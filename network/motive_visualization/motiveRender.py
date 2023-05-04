#!/usr/bin/env python3

import pyray as rl
from motiveParser import *
import sys
import random

def vector3_frametime():
    ft = rl.get_frame_time()
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

def _camera_project(pos, matView, matPerps):
    temp = rl.Vector4()
    result = rl.Vector4()
    temp.x=matView.m0*pos.x+matView.m4*pos.y+matView.m8*pos.z+matView.m12
    temp.y=matView.m1*pos.x+matView.m5*pos.y+matView.m9*pos.z+matView.m13
    temp.z=matView.m2*pos.x+matView.m6*pos.y+matView.m10*pos.z+matView.m14
    temp.w=matView.m3*pos.x+matView.m7*pos.y+matView.m11*pos.z+matView.m15
    result.x=matPerps.m0*temp.x+matPerps.m4*temp.y+matPerps.m8*temp.z+matPerps.m12*temp.w
    result.y=matPerps.m1*temp.x+matPerps.m5*temp.y+matPerps.m9*temp.z+matPerps.m13*temp.w
    result.z=matPerps.m2*temp.x+matPerps.m6*temp.y+matPerps.m10*temp.z+matPerps.m14*temp.w
    result.w=-temp.z
    if result.w != 0.0:
        result.w=(1.0/result.w)/.75    # TODO fudge of .75 WHY???
        # Perspective division
        result.x*=result.w
        result.y*=result.w
        result.z*=result.w
        return result
    else:
        return result
        #result.x = result.y = result.z = result.w

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

    def project(self, x, y, z):
        view = rl.matrix_look_at(self.cam.position, self.cam.target, self.cam.up)
        # aspect = self.width / self.height;
        aspect = rl.get_screen_width() / rl.get_screen_height();

        
        perps = rl.matrix_perspective(self.cam.fovy, aspect, 0.01, 1000.0)
        # return _camera_project(rl.vector3(x,y,z), view, perps)
        return _camera_project(rl.Vector3(x,y,z), view, perps)
       
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

    def parseFile(self, motive_csv_name):
        self.mp = motiveParser(motive_csv_name)

    def run(self, motive_csv_name):
        # Parse file for box poses 
        self.parseFile(motive_csv_name)
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

        frame_timer = 0
        frame = 0
        index_textsize = 18
        is_paused = False
        is_showing_help = False
        frameskip = 100
        playback_speed = 1

        while not rl.window_should_close():
            # Manage Time
            if not is_paused:
                frame_timer = frame_timer + rl.get_frame_time()

            # Pause/Resume on Space 
            if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE):
                if is_paused:
                    is_paused = False
                else:
                    is_paused = True

            # Pause/Resume on Space 
            if rl.is_key_pressed(rl.KeyboardKey.KEY_H):
                if is_showing_help:
                    is_showing_help = False
                else:
                    is_showing_help = True

            # Go back in time
            if rl.is_key_pressed(rl.KeyboardKey.KEY_LEFT):
                frame -= frameskip
                if frame < 0:
                    frame = 0
            # Go forward in time
            if rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT):
                frame += frameskip

            # Increase speed factor
            if rl.is_key_pressed(rl.KeyboardKey.KEY_DOWN):
                playback_speed -= 0.5
                if playback_speed < 0:
                    playback_speed = 0
            # Decrease speed factor
            if rl.is_key_pressed(rl.KeyboardKey.KEY_UP):
                playback_speed += 0.5

            # Reset on Key Z
            if rl.is_key_pressed(rl.KeyboardKey.KEY_Z):
                frame = 0
                frame_timer = 0

            # Reparse on Key R 
            if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
                print(f"REPARSING FILE {motive_csv_name}!")
                self.parseFile(motive_csv_name)
                frame = 0
                frame_timer = 0

            # Update current frame
            # TODO: This is not correct on frame-locked screens, you need to
            #       Explicitly calculate what frame number we are on, based on 
            #       real time, not just be dumb and count like this.
            if frame_timer * playback_speed > (float(self.mp.framerate) / time_scale):
                frame = frame + 1
                frame_timer = 0
            if frame > int(self.mp.framecount)-1:
                frame = 0

            rl.begin_drawing()
            rl.clear_background(rl.WHITE)
            cam.update()
            cam.beginRenderTo()
            # Render 3D grid and Markers
            rl.draw_grid(20, 1.0)
            projections = []
            projection_idxs = []
            idx = 0
            for color, marker in zip(self.markerColors, self.mp.markers[frame]):
                mpos = rl.Vector3(marker.pos[0] / boxworld_scale, 
                                  marker.pos[1] / boxworld_scale,
                                  marker.pos[2] / boxworld_scale)
                if mpos.x != 0 and mpos.y != 0 and mpos.z != 0:
                    rl.draw_cube(mpos, box_size, box_size, box_size, color);
                    projections.append(cam.project(mpos.x, mpos.y, mpos.z))
                    projection_idxs.append(idx)
                idx += 1

            cam.endRenderTo()

            # Render 2D
            hsw = rl.get_screen_width() / 2.0
            hsh = rl.get_screen_height() / 2.0
            for i, p in zip(projection_idxs, projections):
                if p.w > 0:
                    u = int(hsw+(p.x*hsw))
                    v = int(hsh-(p.y*hsh))
                    rl.draw_text(f"{i}", u-10, v+10, index_textsize, rl.BLACK)
                    
            helptext = """
            ESC \t\t To Quit
            WASD \t\t To move horizontally
            QE \t\t To move vertically
            H \t\t Toggle help menu
            SPACE \t\t pause/unpause
            R \t\t Reparse file
            Z \t\t Jump to frame 0
            LEFT \t\t jump 100 frames forward
            RIGHT \t\t jump 100 frames backward
            UP \t\t Speed up by 0.5x
            DOWN \t\t Speed down by 0.5x
            """
            if is_showing_help:
                rl.draw_text(helptext, 2, 200, 20, rl.BLACK)

            rl.draw_text(self.mp.getHeaderString(), 2, 2, 20, rl.BLACK)
            camtext = f"camPos: [{cam.cam.position.x:.3f} {cam.cam.position.y:.3f} {cam.cam.position.z:.3f}]"
            rl.draw_text(f"[PRESS H FOR HELP]", 2, rl.get_screen_height() - 130, 20, rl.BLACK)
            rl.draw_text(f"Playback Speed: {playback_speed}x", 2, rl.get_screen_height() - 88, 20, rl.BLACK)
            rl.draw_text(f"Frame: {frame}", 2, rl.get_screen_height() - 66, 20, rl.BLACK)
            timestamp = float(self.mp.frametimes[frame-1])
            rl.draw_text(f"Timestamp: {timestamp:.2f} s", 2, rl.get_screen_height() - 44, 20, rl.BLACK)
            rl.draw_text(camtext, 2, rl.get_screen_height() - 22, 20, rl.BLACK)
            rl.end_drawing()

        # Window Exit Context
        rl.close_window()
# boxPoseRender

def main():
    bpr = boxPoseRender(fps = -1)
    if len(sys.argv) < 1:
        print("ERROR! Expected to get path to motive csv file!")
        return 1
    bpr.run(sys.argv[1])

if __name__ == "__main__":
    main()
