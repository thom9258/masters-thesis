# Documentation:
# https://electronstudio.github.io/raylib-python-cffi/
# Example code:
# https://electronstudio.github.io/raylib-python-cffi/pyray.html?highlight=white
# Camera rotation
# TODO: Do this shit:
# https://learnopengl.com/Getting-started/Camera
 
import raylib
import pyray as rl
from motiveParser import *
from rlHelper import *
import random

class flightCamera():
    def __init__(self, pos=[-10.0, 18.0, 0], fov=45.0):
        tdir = rl.Vector3(1.0, -1.0, 0.0)
        v3pos = rl.Vector3(pos[0], pos[1], pos[2])
        target = rl.vector3_add(v3pos, tdir)
        up=[0.0, 1.0, 0.0]
        self.cam = rl.Camera3D(pos, target, up, fov, 0)


    def setPose(self, pos=[0, 0, 0]):
        self.cam.position = pos


    def update(self):
        # Flight controls
        wasd = rl.vector3_multiply(wasd_inputaxis(), vector3_val(5))
        qe = rl.vector3_multiply(qe_inputaxis(), vector3_val(5))
        wasd_dt = rl.vector3_multiply(wasd, vector3_frametime())
        qe_dt = rl.vector3_multiply(qe, vector3_frametime())
        moveset = wasd_dt
        moveset.y = qe_dt.y
        self.cam.position = rl.vector3_add(self.cam.position, moveset)
        self.cam.target = rl.vector3_add(self.cam.target, moveset)

       
    def beginRenderTo(self):
        rl.update_camera(self.cam)
        rl.begin_mode_3d(self.cam)


    def endRenderTo(self):
        rl.end_mode_3d()
# flightCamera

class boxPoseRender():
    def __init__(self, w=1200, h=800, fps=60, name="Box Pose Renderer"):
        self.window_name = name
        self.window_w = w
        self.window_h = h
        self.fps = fps


    def run(self, motive_csv_name):
        # Parse file for box poses 
        self.mp = motiveParser(motive_csv_name)
        self.mp.printHeader()
        self.markerColors = []
        for i in range(self.mp.markercount):
            r = random.uniform(0, 1)
            g = random.uniform(0, 1)
            b = random.uniform(0, 1)
            # print(f"generated random color (i) =  {r}, {g}, {b}")
            self.markerColors.append(rl.color_from_normalized([r,g,b,1]))

        # Window Init Context
        cam = flightCamera()
        rl.init_window(self.window_w, self.window_h, self.window_name)
        if self.fps > -1:
            rl.set_target_fps(self.fps)

        # NOTE: Tuneable Magic numbers
        boxworld_scale = 200
        box_size = 0.1
        time_scale = 40000

        timer = 0
        real_time = 0
        recording_time = 0
        frame = 0

        while not rl.window_should_close():
            # Manage Time
            timer = timer + raylib.GetFrameTime()
            real_time = real_time + raylib.GetFrameTime()
            # TODO: This is not correct on frame-locked screens, you need to
            #       Explicitly calculate what frame number we are on, based on 
            #       real time, not just be dumb and count like this.
            if timer > (float(self.mp.framerate) / time_scale):
                frame = frame + 1
                timer = 0
            if frame > int(self.mp.framecount)-1:
                frame = 0
                real_time = 0

            # Reset on Key R
            if rl.is_key_down(rl.KeyboardKey.KEY_R):
                frame = 0
                real_time = 0
                timer = 0

            rl.begin_drawing()
            rl.clear_background(rl.WHITE)
            cam.update()
            cam.beginRenderTo()
            # Render 3D grid and Markers
            rl.draw_grid(20, 1.0)
            for color, marker in zip(self.markerColors, self.mp.markers[frame]):
                mpos = rl.Vector3(marker.pos[0] / boxworld_scale, 
                                  marker.pos[1] / boxworld_scale,
                                  marker.pos[2] / boxworld_scale)
                rl.draw_cube(mpos, box_size, box_size, box_size, color);
            cam.endRenderTo()

            # Render 2D
            rl.draw_text(self.mp.getHeaderString(), 2, 2, 20, rl.BLACK)
            camtext = f"camPos: [{cam.cam.position.x:.3f} {cam.cam.position.y:.3f} {cam.cam.position.z:.3f}]"
            rl.draw_text(f"Frame: {frame}", 2, self.window_h - 88, 20, rl.BLACK)
            rl.draw_text(f"Real Time: {real_time:.2f} s", 2, self.window_h - 66, 20, rl.BLACK)
            recording_time = float(self.mp.frametimes[frame-1])
            rl.draw_text(f"Recording Time: {recording_time:.2f} s", 2, self.window_h - 44, 20, rl.BLACK)
            rl.draw_text(camtext, 2, self.window_h - 22, 20, rl.BLACK)
            rl.end_drawing()

        # Window Exit Context
        rl.close_window()
# boxPoseRender

if __name__ == "__main__":
    print("TEST:")
    bpr = boxPoseRender(fps = -1)
    # path = "./tjens18_test_movement01_xyz_marker_track.csv"
    # path = "../datasets/grasp_dataset/tjens18_index_30s_1.csv"
    path = "../datasets/grasp_dataset/tjens18grasptest2_15s.csv"
    bpr.run(path)
