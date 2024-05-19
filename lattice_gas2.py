import argparse
import datetime
import math
import random
import time
import cupy as xp
from cupyx.scipy import signal
from cupyx.scipy.ndimage import affine_transform
#import numpy as xp
#from scipy import signal
import cv2
from PIL import Image

from typing import Any, Dict, List, Optional

INDEX_0 = 0
INDEX_1 = 1
INDEX_2 = 2
INDEX_3 = 3
INDEX_4 = 4
INDEX_5 = 5
INDEX_6 = 6

DIR_0 = 1 << INDEX_0
DIR_1 = 1 << INDEX_1
DIR_2 = 1 << INDEX_2
DIR_3 = 1 << INDEX_3
DIR_4 = 1 << INDEX_4
DIR_5 = 1 << INDEX_5
DIR_6 = 1 << INDEX_6

def send_e(cells: Any) -> Any:
    return xp.roll(cells, 1, axis = 1)
def send_ne(cells: Any) -> Any:
    return xp.roll(xp.roll(cells, 1, axis = 1), -1, axis = 0)
def send_n(cells: Any) -> Any:
    return xp.roll(cells,                       -1, axis = 0)
def send_nw(cells: Any) -> Any:
    return xp.roll(xp.roll(cells, -1, axis = 1), -1, axis = 0)
def send_w(cells: Any) -> Any:
    return xp.roll(cells, -1, axis = 1)
def send_sw(cells: Any) -> Any:
    return xp.roll(xp.roll(cells, -1, axis = 1), 1, axis = 0)
def send_s(cells: Any) -> Any:
    return xp.roll(cells,                        1, axis = 0)
def send_se(cells: Any) -> Any:
    return xp.roll(xp.roll(cells, 1, axis = 1), 1, axis = 0)

#AVERAGE = xp.ones((32, 32), dtype = xp.float16) / 1024.0
AVERAGE = xp.ones((8, 8), dtype = xp.float16) / 64.0
#AVERAGE = xp.ones((4, 4), dtype = xp.float16) / 16.0

def m_x(index: int) -> Any:
    u'''
    与えられた位置の粒子の X 方向運動量を返す。
    '''
    if index == 0:
        return 0
    return xp.cos(2 * xp.pi * (index - 1) / 6)
def m_y(index: int) -> Any:
    u'''
    与えられた位置の粒子の Y 方向運動量を返す。
    '''
    if index == 0:
        return 0
    return xp.sin(2 * xp.pi * (index - 1) / 6)
class Ruler(object):
    def __init__(self) -> None:
        u'''運動量・粒子数を保存する衝突ルール群を生成する
        '''
        rules = dict()
        for state1 in range(2**7):
            momentum1 = self._momentum_of(state1)
            state2s = []
            for state2 in range(2**7):
                momentum2 = self._momentum_of(state2)
                if self._eq_momentum(momentum1, momentum2) and \
                   self._eq_n_particles(state1, state2):
                    state2s.append(state2)
            if len(state2s) > 1:
                rules[state1] = state2s
        self._rules = rules
        print('{} special rules'.format(len(self._rules)))
    def _momentum_of(self, state: int) -> tuple[float, float]:
        u'''
        与えられた状態の運動量の合計を返す。
        '''
        sum_x = 0
        sum_y = 0
        for n in range(7):
            if state & (1 << n) != 0:
                sum_x += m_x(n)
                sum_y += m_y(n)
        return (sum_x, sum_y)
    def _eq_momentum(self,
                     momentum_a: tuple[float, float],
                     momentum_b: tuple[float, float]) -> bool:
        u'''
        運動量 momentum_a, momentum_b が一致するか調べる。
        '''
        ma_x, ma_y = momentum_a
        mb_x, mb_y = momentum_b
        return True if (ma_x - mb_x)**2 + (ma_y - mb_y)**2 < 0.0001 else False
    def _n_particles(self, state: int) -> int:
        u'''
        与えられた状態の粒子数を返す。
        '''
        return state.bit_count()
    def _eq_n_particles(self, a: int, b: int) -> bool:
        u'''
        状態 a, b で粒子数が一致するか調べる。
        '''
        return self._n_particles(a) == self._n_particles(b)
    def _pick_random_rule(self) -> Dict[int, int]:
        u'''既知の衝突ルールからランダムに選択する。
        元と同じままでよいなら戻り値から除外する。
        '''
        #rule = xp.zeros(2 ** 7, dtype = xp.uint8)
        rule = dict()
        for state1, state2s in self._rules.items():
            state2 = state2s[random.randrange(len(state2s))]
            if state1 == state2:
                continue
            rule[state1] = state2
        return rule
    def apply_rule(self, cells: Any) -> Any:
        u'''ランダムに衝突ルールを選択・適用する
        '''
        cells2 = cells.copy()
        rule = self._pick_random_rule()
        # self.dump_rule(rule)
        for state1, state2 in rule.items():
            cells2 = xp.where(
                cells == state1, xp.asanyarray(state2), cells2)
        return cells2
    def _char(self, state: int, index: int) -> str:
        u'''
        デバッグ用：与えられた状態 state の位置 index に
        粒子が存在するなら '*' を返す。
        なければその位置に応じたキャラクタを返す。
        '''
        if state & (1 << index):
            return '*'
        if index == INDEX_0: return '+'
        if index == INDEX_1: return '-'
        if index == INDEX_2: return '/'
        if index == INDEX_3: return '\\'
        if index == INDEX_4: return '-'
        if index == INDEX_5: return '/'
        return '\\'
    def _dump_rule_entry(self, st1: int, st2: int) -> None:
        u'''デバッグ用：変換ルールの１つをテキスト表示する
        '''
        fmt = '''
        # 0b{:07b} 0b{:07b}
        #  {} {}      {} {}
        # {} {} {} -> {} {} {}
        #  {} {}      {} {}
        '''
        print(fmt.format(
            st1, st2,
            # line 1
            self._char(st1, INDEX_3), self._char(st1, INDEX_2),
            self._char(st2, INDEX_3), self._char(st2, INDEX_2),
            # line 2
            self._char(st1, INDEX_4), self._char(st1, INDEX_0),
            self._char(st1, INDEX_1),
            self._char(st2, INDEX_4), self._char(st2, INDEX_0),
            self._char(st2, INDEX_1),
            # line 3
            self._char(st1, INDEX_5), self._char(st1, INDEX_6),
            self._char(st2, INDEX_5), self._char(st2, INDEX_6)))
    def dump_rule(self, rule: Dict[int, int]) -> None:
        u'''デバッグ用：変換ルールをテキスト表示する
        '''
        for state1, state2 in rule.items():
            self._dump_rule_entry(state1, state2)
class Field(object):
    def __init__(self, width: int, height: int, rule: Ruler) -> None:
        self._width = width
        self._height = height
        self._rule = rule
        self._cells = xp.zeros((height, width), dtype = xp.uint8)
        self._mesh_x, self._mesh_y = xp.meshgrid(
            xp.arange(width), xp.arange(height))
        self._cylinder = (
            (self._mesh_x - self._width / 4) **2 +
            (self._mesh_y - self._height / 2) ** 2 * (3 / 4) < (self._height / 20) ** 2)
    def init_random(self, dens: float) -> None:
        self._cells = (
            xp.where(xp.random.rand(self._height, self._width) * 100 <= dens,
                     xp.uint8(DIR_0), xp.uint8(0)) +
            xp.asanyarray(xp.uint8(DIR_1)) +
            xp.where(xp.random.rand(self._height, self._width) * 100 <= dens,
                     xp.uint8(DIR_2), xp.uint8(0)) +
            xp.where(xp.random.rand(self._height, self._width) * 100 <= dens,
                     xp.uint8(DIR_3), xp.uint8(0)) +
            xp.where(xp.random.rand(self._height, self._width) * 100 <= dens,
                     xp.uint8(DIR_4), xp.uint8(0)) +
            xp.where(xp.random.rand(self._height, self._width) * 100 <= dens,
                     xp.uint8(DIR_5), xp.uint8(0)) +
            xp.where(xp.random.rand(self._height, self._width) * 100 <= dens,
                     xp.uint8(DIR_6), xp.uint8(0)))
        self._cells[self._cylinder] = 0
    def flow(self) -> None:
        cells2 = xp.where(
            (self._mesh_y % 2) == 1,
            ((self._cells & DIR_0) +
             send_e(self._cells & DIR_1) +
             send_ne(self._cells & DIR_2) +
             send_n(self._cells & DIR_3) +
             send_w(self._cells & DIR_4) +
             send_s(self._cells & DIR_5) +
             send_se(self._cells & DIR_6)),
            ((self._cells & DIR_0) +
             send_e(self._cells & DIR_1) +
             send_n(self._cells & DIR_2) +
             send_nw(self._cells & DIR_3) +
             send_w(self._cells & DIR_4) +
             send_sw(self._cells & DIR_5) +
             send_s(self._cells & DIR_6)))
        self._cells = cells2
    def collision(self) -> None:
        #print('    apply_rule')
        self._cells = self._rule.apply_rule(self._cells)
        #print('    cylindrical shape boundry')
        self._cells = xp.where(
            self._cylinder,
            xp.where((self._cells & DIR_1) != 0,
                     xp.asanyarray(xp.uint8(DIR_4)),
                     xp.asanyarray(xp.uint8(0))) +
            xp.where((self._cells & DIR_2) != 0,
                     xp.asanyarray(xp.uint8(DIR_5)),
                     xp.asanyarray(xp.uint8(0))) +
            xp.where((self._cells & DIR_3) != 0,
                     xp.asanyarray(xp.uint8(DIR_6)),
                     xp.asanyarray(xp.uint8(0))) +
            xp.where((self._cells & DIR_4) != 0,
                     xp.asanyarray(xp.uint8(DIR_1)),
                     xp.asanyarray(xp.uint8(0))) +
            xp.where((self._cells & DIR_5) != 0,
                     xp.asanyarray(xp.uint8(DIR_2)),
                     xp.asanyarray(xp.uint8(0))) +
            xp.where((self._cells & DIR_6) != 0,
                     xp.asanyarray(xp.uint8(DIR_3)),
                     xp.asanyarray(xp.uint8(0))),
            self._cells)
    def get_current_bgr_image(self, scale: int) -> Any:
        # X 方向, Y 方向への流速を計算する
        ux = signal.convolve2d(
            (xp.where((self._cells & DIR_1) != 0,
                      xp.asanyarray(m_x(INDEX_1)), xp.asanyarray(0)) +
             xp.where((self._cells & DIR_2) != 0,
                      xp.asanyarray(m_x(INDEX_2)), xp.asanyarray(0)) +
             xp.where((self._cells & DIR_3) != 0,
                      xp.asanyarray(m_x(INDEX_3)), xp.asanyarray(0)) +
             xp.where((self._cells & DIR_4) != 0,
                      xp.asanyarray(m_x(INDEX_4)), xp.asanyarray(0)) +
             xp.where((self._cells & DIR_5) != 0,
                      xp.asanyarray(m_x(INDEX_5)), xp.asanyarray(0)) +
             xp.where((self._cells & DIR_6) != 0,
                      xp.asanyarray(m_x(INDEX_6)), xp.asanyarray(0))),
            AVERAGE, mode = 'same', boundary = 'wrap')[::scale,::scale]
        uy = signal.convolve2d(
            (xp.where((self._cells & DIR_1) != 0,
                      xp.asanyarray(m_y(INDEX_1)), xp.asanyarray(0)) +
             xp.where((self._cells & DIR_2) != 0,
                      xp.asanyarray(m_y(INDEX_2)), xp.asanyarray(0)) +
             xp.where((self._cells & DIR_3) != 0,
                      xp.asanyarray(m_y(INDEX_3)), xp.asanyarray(0)) +
             xp.where((self._cells & DIR_4) != 0,
                      xp.asanyarray(m_y(INDEX_4)), xp.asanyarray(0)) +
             xp.where((self._cells & DIR_5) != 0,
                      xp.asanyarray(m_y(INDEX_5)), xp.asanyarray(0)) +
             xp.where((self._cells & DIR_6) != 0,
                      xp.asanyarray(m_y(INDEX_6)), xp.asanyarray(0))),
            AVERAGE, mode = 'same', boundary = 'wrap')[::scale,::scale]
        # 回転を計算
        vorticity = signal.convolve2d(
            (xp.roll(ux, -1, axis=0) - xp.roll(ux, 1, axis=0)) -
            (xp.roll(uy, -1, axis=1) - xp.roll(uy, 1, axis=1)),
            AVERAGE, mode = 'same', boundary = 'wrap')
        gh = self._height // scale
        gw = self._width // scale
        img = xp.zeros([gh, gw, 3], dtype = xp.uint8)
        # color B ... -X 方向への流速
        img[:,:,0] = (
            (xp.max(ux) - ux) * 255 / (xp.max(ux) - xp.min(ux))
        ).astype(xp.uint8)
        # color G ... 右回転
        img[:,:,1] = xp.where(
            vorticity < 0,
            vorticity * 255 / xp.min(vorticity),
            xp.asanyarray(0)).astype(xp.uint8)
        # color R ... 左回転
        img[:,:,2] = xp.where(
            vorticity > 0,
            vorticity * 255 / xp.max(vorticity),
            xp.asanyarray(0)).astype(xp.uint8)
        # 円柱のある場所を塗りつぶす
        img[self._cylinder[::scale,::scale],:] = 127
        img = affine_transform(img, xp.array([
            [1, 0, 0],
            [0, math.sqrt(3) / 2, 0],
            [0, 0, 1]]))
        #print('  asnumpy')
        return xp.asnumpy(img)
class Animation(object):
    def __init__(self, width: int, height: int,
                 fps: float, outfile: Optional[str] = None) -> None:
        if not outfile:
            date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            outfile = date + '.mp4'
        fmt = cv2.VideoWriter_fourcc(*'avc1')
        self._writer = cv2.VideoWriter(outfile, fmt, fps,
                                       (width, height))
    def capture(self, bgr: Any) -> None:
        self._writer.write(bgr)
    def make_gif(self) -> None:
        self._writer.release()
def print_elapsed_time(start_time: float, msg: str) -> None:
    current_time = time.time()
    elapse = current_time - start_time
    print("{:02}:{:02}:{:02}.{:09} - {}".format(
        int(elapse / 3600),
        int(elapse / 60) % 60,
        int(elapse) % 60,
        int(elapse * 1000000000) % 1000000000,
        msg))
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type = int, default = 2048,
                        help = 'Field width (default 2048)')
    parser.add_argument('--height', type = int, default = 1024,
                        help = 'Field height (default 1024)')
    parser.add_argument('--scale', type = int, default = 2,
                        help = 'Magnify cell size (default 2)')
    parser.add_argument('--dens', type = float, default = 10,
                        help = 'Density parameter from 0 to 100 (default 10)')
    parser.add_argument('--loop', type = int, default = 10000,
                        help = 'loop count (default 10000)')
    parser.add_argument('--skip', type = int, default = 10,
                        help = 'skip generating image (default 10 times)')
    parser.add_argument('--skip-first', type = int, default = 0,
                        help = 'skip first iteration (default 0 times)')
    parser.add_argument('--animation', action = 'store_true',
                        help = 'animation')
    args = parser.parse_args()
    width = args.width
    height = args.height
    scale = args.scale
    dens = args.dens
    loop = args.loop
    skip = args.skip
    skip_first = args.skip_first
    pool = xp.cuda.MemoryPool(xp.cuda.malloc_managed)
    xp.cuda.set_allocator(pool.malloc)
    ruler = Ruler()
    field = Field(width, height, ruler)
    print('initializing random field')
    field.init_random(dens)
    print('initialized random field')
    waiting = 10
    if args.animation:
        bgr_img = field.get_current_bgr_image(scale)
        bgr_width = bgr_img.shape[1]
        bgr_height = bgr_img.shape[0]
        animation = Animation(bgr_width, bgr_height, 100, outfile = 'fhp.mp4')
    start_time = time.time()
    for n in range(loop):
        #print('  flow')
        field.flow()
        #print('  collision')
        field.collision()
        if skip_first and n <= skip_first:
            print_elapsed_time(start_time, '{} / {}'.format(n, loop))
            continue
        if skip < 2 or n % skip == 0:
            print_elapsed_time(start_time, '{} / {}'.format(n, loop))
            #print('  bgr_image')
            bgr_img = field.get_current_bgr_image(scale)
            #print('  show')
            cv2.imshow("Ceullular Automata", bgr_img)
            key = cv2.waitKey(waiting)
            if key == ord("+"):
                waiting = max(10, waiting//2)
            if key == ord("-"):
                waiting = min(1000, waiting*2)
            if key == ord("q"):
                break
            if args.animation:
                #print('  animation')
                animation.capture(bgr_img)
    if args.animation:
        print('dumping mp4 animation...')
        animation.make_gif()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
