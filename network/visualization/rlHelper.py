import raylib
import pyray as rl

def vector3_frametime():
    ft = raylib.GetFrameTime()
    return rl.Vector3(ft,ft,ft)

def vector3_val(f):
    return rl.Vector3(f,f,f)

def wasd_inputaxis():
    inputaxis = rl.Vector3(0,0,0)
    if rl.is_key_down(rl.KeyboardKey.KEY_W):
        inputaxis.x = inputaxis.x + 1
    if rl.is_key_down(rl.KeyboardKey.KEY_S):
        inputaxis.x = inputaxis.x - 1
    if rl.is_key_down(rl.KeyboardKey.KEY_D):
        inputaxis.z = inputaxis.z + 1
    if rl.is_key_down(rl.KeyboardKey.KEY_A):
        inputaxis.z = inputaxis.z - 1
    #print(f"wasd input: {inputaxis.x} {inputaxis.z}")
    return inputaxis

def qe_inputaxis():
    inputaxis = rl.Vector3(0,0,0)
    if rl.is_key_down(rl.KeyboardKey.KEY_Q):
        inputaxis.y = inputaxis.y + 1
    if rl.is_key_down(rl.KeyboardKey.KEY_E):
        inputaxis.y = inputaxis.y - 1
    #print(f"qe input: {inputaxis.x}")
    return inputaxis



