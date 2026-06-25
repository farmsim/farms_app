""" Hindlimb simulator """


import collections
import math
from functools import cache


import numpy as np
import OpenGL.GL as GL  # type: ignore
from farms_core import pylog
from imgui_bundle import imgui, implot
from imgui_bundle import portable_file_dialogs as pfd
from farms_app.extensions.base import CustomExtension
from farms_core.io.yaml import read_yaml
import hindlimb_locomotion
from hindlimb_locomotion.tools import  state_to_limbmodel, extract_vars
from hindlimb_locomotion.tools import calcFrequency_simple as calcFrequency
from datetime import datetime
import time


@cache
def thip(a): return -1.0*(a-math.pi/2.0)

@cache
def tknee(a): return a+math.pi

@cache
def tankle(a): return -1.0*(a-math.pi)

@cache
def transfer_function(x):
    return (x+50)/50. if -50 < x < 0 else 0 if x < -50 else 1.0

@cache
def mn_act_function(x): return 1.0 if x > 1.0 else 0 if x < 0.0 else x


def tint(c, f): return [v + (1.0 - v) * f for v in c]


def shade(c, f): return [v * (1.0 - f) for v in c]

class PhaseStim:
    def __init__(self, interval = 1.0, duration = 0.01):
        self.interval = interval
        self.duration = duration
        self.phase_stim_time = -100.0



farms_info = {
    "name": "Mouse Hindlimb",
    "stage": "",
    "version": (0, 0, 1),
    "author": "FARMS"
}


class MouseHindlimb2DExtension(CustomExtension):
    """ 2D Mouse Simulation """

    def __init__(self):
        self.show_window = True
        self.window_name = "Hindlimb"
        self._io = imgui.get_io()
        self.sim = None
        self.yaml_config_path = None
        self.performance_warnings = []

    def get_name(self) -> str:
        return "Mouse Hindlimb"

    def draw_limbs(self):
        if hasattr(self.sim, 'm') and self.sim.m is not None and implot.begin_plot("Limb Model", (-1, 400)):
            # Set up plot limits based on model position
            m = self.sim.m
            x_center = m.torso_center.x if hasattr(m, 'torso_center') else m.hip.as_array()[0]

            # Dynamic window following the model
            plot_width = self.sim.distance_x*0.5
            axes_flags = implot.AxisFlags_.lock | implot.AxisFlags_.no_grid_lines | implot.AxisFlags_.no_tick_marks
            implot.setup_axes("X Position (m)", "Height (m)", y_flags=axes_flags)
            implot.setup_axis_limits(implot.ImAxis_.x1, x_center - plot_width/2, x_center + plot_width/2)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, 0.1)
            implot.setup_axis(implot.ImAxis_.y1, flags=implot.AxisFlags_.lock)
            implot.setup_axis(implot.ImAxis_.x1, flags=implot.AxisFlags_.auto_fit)

            # Draw ground surface
            if hasattr(self.sim, 'sol'):
                x_ground = np.linspace(x_center - plot_width/2, x_center + plot_width/2, 100)
                y_ground = np.array([self.sim.sol.get_surface_height(x) for x in x_ground])
                implot.plot_line("Ground", x_ground, y_ground)

            # Draw foot trajectory history
            if len(self.sim.history) > 1:
                foot_trail_x = []
                foot_trail_y = []
                for i, (m_hist, _, fc, _) in enumerate(self.sim.history):
                    if hasattr(m_hist, 'foot') and len(m_hist.foot) > 0:
                        foot_pos = m_hist.foot[0].as_array()
                        foot_trail_x.append(foot_pos[0])
                        foot_trail_y.append(foot_pos[1])

                if len(foot_trail_x) > 1:
                    implot.plot_line("Foot Trail", np.array(foot_trail_x), np.array(foot_trail_y))

            # Get current model joint positions
            if hasattr(m, 'hip') and hasattr(m, 'knee') and hasattr(m, 'ankle') and hasattr(m, 'foot'):
                # Extract coordinates
                hip = np.array(m.hip.as_array())
                pelvis = np.array(m.pelvis[0].as_array()) if hasattr(m, 'pelvis') else hip

                # Draw body/torso
                body_x = [pelvis[0], hip[0]]
                body_y = [pelvis[1], hip[1]]
                implot.plot_line("Body", np.array(body_x), np.array(body_y))

                # Draw both legs
                for leg_idx in range(min(2, len(m.knee))):
                    knee = np.array(m.knee[leg_idx].as_array())
                    ankle = np.array(m.ankle[leg_idx].as_array())
                    foot = np.array(m.foot[leg_idx].as_array())

                    # Hip to knee segment
                    thigh_x = [hip[0], knee[0]]
                    thigh_y = [hip[1], knee[1]]
                    leg_name = f"Thigh_L" if leg_idx == 0 else f"Thigh_R"
                    implot.plot_line(leg_name, np.array(thigh_x), np.array(thigh_y))

                    # Knee to ankle segment (shank)
                    shank_x = [knee[0], ankle[0]]
                    shank_y = [knee[1], ankle[1]]
                    leg_name = f"Shank_L" if leg_idx == 0 else f"Shank_R"
                    implot.plot_line(leg_name, np.array(shank_x), np.array(shank_y))

                    # Ankle to foot segment
                    foot_x = [ankle[0], foot[0]]
                    foot_y = [ankle[1], foot[1]]
                    leg_name = f"Foot_L" if leg_idx == 0 else f"Foot_R"
                    implot.plot_line(leg_name, np.array(foot_x), np.array(foot_y))

                    # Draw joints as scatter points
                    joint_x = [hip[0], knee[0], ankle[0]]
                    joint_y = [hip[1], knee[1], ankle[1]]
                    joint_name = f"Joints_L" if leg_idx == 0 else f"Joints_R"
                    implot.plot_scatter(f"##{joint_name}", np.array(joint_x), np.array(joint_y))

            # Show contact forces if available
            if len(self.sim.state_hist) > 0 and len(self.sim.state_hist[-1]) > 7:
                current_state = self.sim.state_hist[-1]
                foot_contacts = current_state[6]  # foot contact states
                grf = current_state[7]  # ground reaction forces

                # Draw force vectors at foot contact points
                if hasattr(m, 'foot') and len(foot_contacts) >= 2:
                    for leg_idx in range(min(2, len(m.foot))):
                        if foot_contacts[leg_idx] > 0.5:  # If foot is in contact
                            foot_pos = np.array(m.foot[leg_idx].as_array())
                            # Scale force for visualization
                            force_scale = 0.001
                            force_end_x = foot_pos[0] + grf[0] * force_scale
                            force_end_y = foot_pos[1] + grf[1] * force_scale

                            force_x = [foot_pos[0], force_end_x]
                            force_y = [foot_pos[1], force_end_y]
                            force_name = f"##GRF_L" if leg_idx == 0 else f"GRF_R"
                            implot.plot_line(force_name, np.array(force_x), np.array(force_y))

            implot.end_plot()

    def draw_mn_activity(self):
        if implot.begin_plot("Motoneuron Activity", flags=implot.Flags_.no_title):
            self.sim.mn_reindex = [0, 1, 5, 2, 3, 4, 6]  # IP, GM, BF, VL, TA, SO, GA
            self.sim.mn_names = ['IP', 'GM', 'BF', 'VL', 'TA', 'SO', 'GA']
            implot.setup_axes_limits(0.0, 0.5, 0.0, 7.0)
            if self.sim.state_hist:
                if len(self.sim.state_hist) > 10:
                    time_data = np.array([st[0] for st in self.sim.state_hist])
                    time_data = time_data - time_data[0]  # Normalize to start at 0
                    implot.push_style_var(implot.StyleVar_.fill_alpha, 0.5)
                    # times = np.linspace(0.0, 1.0, 1000)
                    for i_, i in enumerate(self.sim.mn_reindex):
                        if i < len(self.sim.state_hist[0][3]):
                            mn_activity = np.array([mn_act_function(st[3][i]) for st in self.sim.state_hist])
                            implot.plot_shaded(self.sim.mn_names[i_], time_data, i_ + mn_activity, yref=i_)
                            implot.plot_line(self.sim.mn_names[i_], time_data, i_ + mn_activity)
                    implot.pop_style_var()
            implot.end_plot()

    def draw_phase_plots(self):
        plot_flags = implot.Flags_.equal | implot.Flags_.no_title
        axis_flags = implot.AxisFlags_.lock | implot.AxisFlags_.no_grid_lines
        if imgui.begin_child(
                "HipKnee", child_flags=imgui.ChildFlags_.resize_x | imgui.ChildFlags_.resize_y
        ):
            if len(self.sim.state_hist) > 10:
                # Hip-Knee phase plot
                if implot.begin_plot("Hip-Knee Phase", flags=plot_flags):
                    # Extract phase data for both legs
                    hip_angles_l = [thip(st[1]['T11']) for st in self.sim.state_hist]
                    knee_angles_l = [tknee(st[1]['T12']) for st in self.sim.state_hist]
                    hip_angles_r = [thip(st[1]['T21']) for st in self.sim.state_hist]
                    knee_angles_r = [tknee(st[1]['T22']) for st in self.sim.state_hist]

                    # Plot phase trajectories
                    implot.setup_axis(implot.ImAxis_.x1, flags=axis_flags)
                    implot.setup_axis(implot.ImAxis_.y1, flags=axis_flags)
                    implot.setup_axes_limits(0.0, 180.0, 0.0, 180.0)
                    implot.plot_line("Left Leg", np.rad2deg(hip_angles_l), np.rad2deg(knee_angles_l))
                    implot.plot_scatter("##Left Leg", np.rad2deg([hip_angles_l[-1]]), np.rad2deg([knee_angles_l[-1]]))
                    implot.plot_line("Right Leg", np.rad2deg(hip_angles_r), np.rad2deg(knee_angles_r))
                    implot.plot_scatter("##Right Leg", np.rad2deg([hip_angles_r[-1]]), np.rad2deg([knee_angles_r[-1]]))

                    implot.end_plot()
        imgui.end_child()
        imgui.same_line()
        # Hip-Ankle phase plot
        if imgui.begin_child(
            "HipAnkle", child_flags=imgui.ChildFlags_.resize_x | imgui.ChildFlags_.resize_y
        ):
            if len(self.sim.state_hist) > 10:
                if implot.begin_plot("Hip-Ankle Phase", flags=plot_flags):
                    ankle_angles_l = [tankle(st[1]['T13']) for st in self.sim.state_hist]
                    ankle_angles_r = [tankle(st[1]['T23']) for st in self.sim.state_hist]
                    implot.setup_axis(implot.ImAxis_.x1, flags=axis_flags)
                    implot.setup_axis(implot.ImAxis_.y1, flags=axis_flags)
                    implot.setup_axes_limits(0.0, 180.0, 0.0, 180.0)
                    implot.plot_line("Left Leg", np.rad2deg(hip_angles_l), np.rad2deg(ankle_angles_l))
                    implot.plot_scatter("##Left Leg", np.rad2deg([hip_angles_l[-1]]), np.rad2deg([ankle_angles_l[-1]]))
                    implot.plot_line("Right Leg", np.rad2deg(hip_angles_r), np.rad2deg(ankle_angles_r))
                    implot.plot_scatter("##Right Leg", np.rad2deg([hip_angles_r[-1]]), np.rad2deg([ankle_angles_r[-1]]))

                    implot.end_plot()
        imgui.end_child()

    def draw_controls(self):
        if imgui.begin_child("LeftPanel", child_flags=imgui.ChildFlags_.resize_x|imgui.ChildFlags_.resize_y):  # width=200, height=fill, border=True
            # Control panel
            if imgui.collapsing_header("Controls"):
                # Speed control
                changed, self.sim.pb_speed = imgui.slider_float("Playback Speed", self.sim.pb_speed, 0.01, 2.0, "%.2fx")

                # Alpha control
                if imgui.button("Alpha +"):
                    self.sim.sol.update_alpha = False
                    alpha = self.sim.sol.get_alpha()
                    self.sim.sol.set_alpha(alpha * 1.1)
                imgui.same_line()
                if imgui.button("Alpha -"):
                    self.sim.sol.update_alpha = False
                    alpha = self.sim.sol.get_alpha()
                    self.sim.sol.set_alpha(alpha / 1.1)
        imgui.end_child()

        imgui.same_line()

        if imgui.begin_child("RightPanel", child_flags=imgui.ChildFlags_.resize_x|imgui.ChildFlags_.resize_y):  # Fill remaining space
            # Status information
            if imgui.collapsing_header("Status", imgui.TreeNodeFlags_.default_open):
                imgui.text(f"Time: {self.sim.t_:.2f}s")
                imgui.text(f"Alpha: {self.sim.sol.get_alpha():.2f}")
                imgui.text(f"Speed: {self.sim.filtered_speed:.2f}")
                imgui.text(f"Frequency: {self.sim.frequency:.2f} Hz")
                imgui.text(f"Stride Length: {self.sim.stride_length*100.0:.2f} cm")
                imgui.text(f"FPS: {self.sim.actual_fps:.2f}")

        imgui.end_child()

    def draw_neural_activity(self):
        if len(self.sim.state_hist) > 10:
            time_data = np.array([st[0] for st in self.sim.state_hist])
            time_data = time_data - time_data[0]  # Normalize to start at 0

            # RG neuron activity
            if implot.begin_plot("Rhythm Generator Activity"):
                rg_f_l = np.array([transfer_function(st[2][0]) for st in self.sim.state_hist])
                rg_e_l = np.array([transfer_function(st[2][2]) for st in self.sim.state_hist])
                v3e_l = np.array([transfer_function(st[2][30]) for st in self.sim.state_hist])

                implot.plot_line("RG_F_L", time_data, rg_f_l)
                implot.plot_line("RG_E_L", time_data, rg_e_l)
                implot.plot_line("V3E_L", time_data, v3e_l)

                implot.end_plot()

    def render_window(self) -> None:
        if imgui.begin_menu_bar():
            if imgui.begin_menu("File"):
                if imgui.menu_item("Open", shortcut="o", p_selected=False)[0]:
                    self.yaml_config_path = pfd.open_file("Load Yaml")
                imgui.end_menu()
            imgui.end_menu_bar()

        if self.yaml_config_path is not None:
            self.sim = HindlimbSimulator(self.yaml_config_path.result()[0])
            self.yaml_config_path = None

        if self.sim:
            # At the start of your render loop
            current_time = time.perf_counter()
            if not hasattr(self, 'last_frame_time'):
                self.last_frame_time = current_time

            frame_dt = current_time - self.last_frame_time
            self.last_frame_time = current_time

            # Calculate how much simulation time to advance
            sim_dt = HindlimbSimulator.dt
            target_sim_time = frame_dt * self.sim.pb_speed  # pb_speed is the playback multiplier

            # Run simulation steps to cover the target simulation time
            accumulated_sim_time = 0.0
            while accumulated_sim_time < target_sim_time:
                a = HindlimbSimulator.a
                self.sim.actual_fps = a * (1.0 / sim_dt) + (1.0 - a) * self.sim.actual_fps
                self.sim.do_iteration(sim_dt)
                accumulated_sim_time += sim_dt

            self.draw_controls()

            self.draw_limbs()

            self.draw_phase_plots()
            imgui.same_line()
            self.draw_neural_activity()

            self.draw_mn_activity()


class HindlimbSimulator:
    """ Hindlimb Simulator """

    dt = 0.0005
    a = 0.05
    dt_remainder = 0.0

    def __init__(self, yaml_config_path):
        " Initial simulation "
        self.sol = hindlimb_locomotion.Solver(yaml_config_path)
        pylog.debug("Initialized Hindlimb Solver")
        self.m = self.sol.get_model()
        self.mn_names = self.sol.get_mn_names()

        self.in_act = self.sol.get_in_act()
        self.reset_state = [self.sol.get_limb_state(), self.sol.get_network_state()]
        self.neuron_names = self.sol.get_neuron_names()


        # rename motoneurons
        for j in range(len(self.mn_names)):
            for i in range(len(self.mn_names[j])):
                for sub in [('BF','ST'),('IP','IL'),('GM','BF'),('_hind',"")]:
                    self.mn_names[j][i] = self.mn_names[j][i].replace(sub[0],sub[1])


        fb_Ibe = ['fbIb_GM', 'fbIb_VL', 'fbIb_SO', 'fbIb_BF', 'fbIb_GA']
        fb_Ia_mon = ['fbIa_GM', 'fbIa_VL', 'fbIa_SO', 'fbIa_BF', 'fbIa_GA', 'fbIa_IP', 'fbIa_TA']
        fb_Ia_rec = ['fbIa_GM_Ia', 'fbIa_VL_Ia', 'fbIa_SO_Ia', 'fbIa_BF_Ia', 'fbIa_GA_Ia', 'fbIa_IP_Ia', 'fbIa_TA_Ia']
        fb_II_RG = ['fbII_TA_InII', 'fbII_IP_InII']
        inII_RG = ['InII_IP_RGF', 'InII_IP_InRGF', 'InII_TA_RGF', 'InII_TA_InRGF']
        fb_Ib_RG = ['fbIb_SO_InRGE', 'fbIb_SO_RGE', 'fbIb_GA_InRGE', 'fbIb_GA_RGE']
        fb_cut_RG = ['fbCutInRGE', 'fbCutRGE']
        fb_II_PF = ['fbII_TA_PFF', 'fbII_IP_PFF', 'fbII_IP_PFSt', 'fbII_TA_PFSt']
        con_V0V_RG = ['inV0VtoRGF', 'V0VtoRGFdiagfh']
        con_V0D_RG = ['V0DtoRGF', 'V0DtoRGFdiagfh']
        con_V3_RG = ['V3EtoRGE', 'V3EtoInE', 'V3FtoRGF', 'V3EtoInE2','V0VtoRGFdiaghf']
        con_shox2_RG = ['V2aHomfh','V2aHomhf']
        desc_LPNs = ['V0VtoRGFdiagfh','V0DtoRGFdiagfh','V2aHomfh','inFH']
        fb_all = fb_Ibe + fb_Ia_mon + fb_Ia_rec + fb_II_RG + fb_Ib_RG + fb_cut_RG + fb_II_PF
        self.variable_vector = fb_all

        self.distance_x = 0.5 #0.5# 0.5 #2 #0.5
        self.t_ = 0.0
        self.prev_t = 0.0
        self.i = 0
        self.pb_speed = 0.2
        self.n_hist = 500

        self.t_last_hist = -1.0
        self.history = collections.deque(maxlen=self.n_hist)

        self.cont_draw_pos = 0.5
        self.hold = False
        self.do_record = False
        self.record_data = False
        self.t0_record = 0.0
        self.frames = list()
        self.i_record = 0
        self.rec_pf = 0
        self.first_update = True
        self.filtered_speed = 0.0


        self.rgf = collections.deque(maxlen=200)
        self.tv = collections.deque(maxlen=200)
        self.uvs = list()
        self.speeds = list()

        self.hist_time = 0.5
        self.state_hist = collections.deque(maxlen=int(self.hist_time/HindlimbSimulator.dt))
        self.N_skip = int(self.hist_time/0.5)
        self.dt_hist = 0.004

        self.rec_hist_dur = 10.

        self.var_off = False
        self.i_step = 0
        self.i_draw = 0

        color_leg1 = [1, 209/255.0, 73/255.0]
        color_leg2 = [198/255.0, 113/255.0, 0]
        color_surface = [1.0, 160/255.0, 0]

        self.dtstr = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        self.n_print = 0
        self.n_video = 0

        colormap = [[0, 0, 0],  # black
                    [230, 159, 0],  # orange
                    [86, 180, 233],  # sky blue
                    [0, 158, 115],  # bluish green
                    [240, 228, 66],  # yellow
                    [0, 114, 178],  # blue
                    [213, 94, 0],  # vermillion
                    [204, 121, 167],
                    [255, 255, 255]]  # reddish purple
        colormap = [[x / 255.0 for x in c] for c in colormap]

        self.bg_color = shade(colormap[8], 0.0)
        self.txt_color = tint(colormap[0], 0.1)
        self.grid_color = tint(colormap[0], 0.1)
        self.color_surface = colormap[7]
        self.color_leg1 = colormap[1]
        #color_leg1 = colormap[1]
        self.color_leg2 = colormap[0]
        self.angle_pl_color = colormap[2]
        self.in_color = colormap[3]
        self.mn_color = colormap[4]

        self.mn_reindex = [0,1,5,2,3,4,6] # IP, GM, BF,  VL,TA, SO, GA
        def hex2rgb(hexstr):
            return tuple(int(hexstr[i:i+2], 16)/255. for i in (0, 2, 4))

        rg_colors = [hex2rgb('ED1C24'),hex2rgb('0B4CA1')]
        self.pf_colors = [hex2rgb('A41C20'),hex2rgb('F07422'),hex2rgb('3B5691'),hex2rgb('4B83B8')]
        self.mn_colors = [rg_colors[0],rg_colors[1],hex2rgb('007D42'),rg_colors[0],rg_colors[1],rg_colors[1],hex2rgb('007D42')]

        self.frequency = -1.0
        self.stride_length = -1.0

        self.model_lw = 0.0005
        self.surface_lw = 0.0005
        self.fps = 120.0

        self.t_transition = None
        self.i_transition = -1
        self.n_yamls = 1

        self.actual_fps = imgui.get_io().framerate
        self.ps_interval = 1.7

        self.do_phase_stim = False
        self.phase_stim_time = -100.0

        self.Ia_scale_factor = 0.05
        self.II_scale_factor = 0.5
        self.Ib_scale_factor = 1.

        self.reset_vec = self.sol.setupVariableVector(self.variable_vector)

    @staticmethod
    def model_to_dict(m,lm):
        coords = { 'IliacCrest': m.pelvis[0].as_array(),
                'Hip': m.hip.as_array(), 'Knee': m.knee[0].as_array(), 'Knee2': m.knee[1].as_array(),
                'Ankle': m.ankle[0].as_array(),'Ankle2': m.ankle[1].as_array(),
                'ToeTip': m.foot[0].as_array(),'ToeTip2': m.foot[1].as_array(),
                'Front_ToeTip': m.flend.as_array()}
        md = {k+'_'+xy: v[i] for k,v in coords.items() for i,xy in enumerate(['x','y']) }
        angles = {'Hip_angle': thip(lm['T11']),'Knee_angle': tknee(lm['T12']),'Ankle_angle': tankle(lm['T13']),
                  'Hip_angle2': thip(lm['T21']),'Knee_angle2': tknee(lm['T22']),'Ankle_angle2': tankle(lm['T23'])}
        md.update(angles)
        return md

    @staticmethod
    def model_to_lines(m):
        return [[m.flend.as_array(), m.shoulder2.as_array(), m.pelvis[0].as_array(),m.pelvis[1].as_array()],
                [m.pelvis[0].as_array(),m.hip.as_array(), m.knee[0].as_array(),
                 m.ankle[0].as_array(), m.foot[0].as_array()],
                [m.pelvis[0].as_array(),m.hip.as_array(), m.knee[1].as_array(), m.ankle[1].as_array(), m.foot[1].as_array()]]

    @staticmethod
    def model_to_lines_ext(m):
        return  [[m.flend.as_array(), m.shoulder2.as_array(), m.pelvis[0].as_array(),m.pelvis[1].as_array()],
                [[m.hip.as_array(), m.knee[0].as_array()],
                [(m.knee[0]+0.2*(m.knee[0]-m.ankle[0])).as_array(), m.ankle[0].as_array()],
                 [(m.ankle[0]+0.2*(m.ankle[0]-m.foot[0])).as_array(), m.foot[0].as_array()]],
                [[m.hip.as_array(), m.knee[1].as_array()],
                [(m.knee[1]+0.2*(m.knee[1]-m.ankle[1])).as_array(), m.ankle[1].as_array()],
                 [(m.ankle[1]+0.2*(m.ankle[1]-m.foot[1])).as_array(), m.foot[1].as_array()]]]

    def do_model_transition(self):
        dphase = 0.1
        f = 0.0
        if self.t_transition != None:
            ts=self.t_transition

            if (self.t_ > ts) and (self.i_transition < self.n_yamls-1):
                if self.t_ <= ts+dphase:
                    f = (self.t_-ts)/(dphase)
                else:
                    f = 1.0
                self.sol.update_alpha = False
                alpha = self.alphas[self.i_transition]*(1.0-f)+self.alphas[self.i_transition+1]*f
                self.sol.set_alpha(alpha)



                #if self.t_ > ts+dphase and self.var_off:
                #    self.var_off = not self.var_off

                uv = [v[self.i_transition+1]*(1.0-f)+v[self.i_transition+2]*f for v in self.variable_values]
                self.sol.updateVariableVector(uv)

    def do_iteration(self, dt):
        start_time = time.perf_counter()
        if self.do_record:
            dt = 1.0/self.fps
        n_steps_f = (dt*self.pb_speed+self.dt_remainder) / self.dt
        n_steps = int(n_steps_f)
        remainder = n_steps_f - n_steps
        self.dt_remainder = (remainder)*self.dt
        for i in range(n_steps):
            if self.sol.step(self.dt):
                lm = state_to_limbmodel(self.sol.get_limb_state())
                gc=self.sol.get_ground_reaction_forces(0)
                #import IPython;IPython.embed()
                self.m = self.sol.get_model()
                mnact = self.sol.get_mn_act()
                fc = [self.sol.get_foot_contact(
                    0), self.sol.get_foot_contact(1)]
                lscond = self.sol.get_sensory_condition()
                Fp = self.sol.get_Fp()
                U=self.sol.get_muscle_torques()
                fh=np.array(self.m.foot[0].as_array())-np.array(self.m.hip.as_array())
                n_fh = np.array([-fh[1],fh[0]])
                fk=np.array(self.m.foot[0].as_array())-np.array(self.m.knee[0].as_array())
                n_fk = np.array([-fk[1],fk[0]])
                fa=np.array(self.m.foot[0].as_array())-np.array(self.m.ankle[0].as_array())
                n_fa = np.array([-fa[1],fa[0]])
                F_ep_muscle = n_fh*U[3]/np.linalg.norm(n_fh)**2+n_fk/np.linalg.norm(n_fk)**2*U[4]+n_fa*U[5]/np.linalg.norm(n_fa)**2
                #mnact_=np.array(mnact)
                #mnact = mnact_*np.random.normal(size=mnact_.shape)
                #mnact = mnact*0.5 + 0.5
                #import IPython;IPython.embed()
                model_lines = self.model_to_lines(self.m)
                self.state_ =[self.t_,
                     lm,
                     self.sol.get_network_state(),
                     mnact[0],
                     mnact[1],
                     self.m.foot[0].as_array(),
                     fc,
                     gc.as_array(),
                     F_ep_muscle,
                     model_lines,
                     [float(self.var_off)],
                     lscond[0].Ia,
                     lscond[0].Ib,
                     lscond[0].II,
                     Fp[0]]
                self.state_hist.append(
                        self.state_
                    )
                if self.record_data and self.i_step %2 == 0:
                    md=self.model_to_dict(self.m,lm)

                    self.rec_hist.append((self.t_, lm,
                                {x[1][1]:x[0] for x in zip(self.sol.get_network_state(),self.neuron_names.items())},
                                fc,
                                gc.as_array(),
                                md,
                                lscond,
                                mnact[0],
                                mnact[1],
                                [float(self.var_off)]))
                a = self.dt*10
                self.filtered_speed = a*lm['Dx'] + (1-a)*self.filtered_speed

                if self.t_last_hist < self.t_ - self.dt_hist:

                    self.history.append((self.m,model_lines, fc, self.var_off))
                    self.t_last_hist = self.t_
                self.t_ += self.dt
                if self.i_step % 10 == 0:
                    self.rgf.append(self.sol.get_network_state()[0])
                    # self.rgf.append(lm['T11'])
                    self.tv.append(self.t_)
                self.i_step += 1

                if self.do_phase_stim:

                    ps_thresh = 0.05
                    ps_dur = 0.025
                    ps_off = 0.0
                    if not hasattr(self,'stim_on'):
                        self.stim_on = False
                    if self.t_ > 1.:
                        #if (mnact[1][3] > ps_thresh) and self.t_ > self.phase_stim_time + ps_interval:
                        if  self.t_ > self.phase_stim_time + self.ps_interval:
                            self.phase_stim_time = self.t_
                        if self.t_ < self.phase_stim_time + ps_dur:
                            if not self.stim_on:
                                self.ps_interval = 1.7 + np.random.normal()*0.1
                                self.stim_on = True
                                self.sol.updateVariableVector([.25] * len(self.reset_vec))
                                print('stim',self.t_)
                        else:
                            if self.stim_on:

                                self.stim_on = False
                                self.sol.updateVariableVector(self.reset_vec)
                                print('stim_off',self.t_)
                    self.var_off = self.stim_on

                #import IPython;IPython.embed()
            if time.perf_counter() > (start_time + 0.9*dt) and not self.do_record:
                break
        #print(time.perf_counter()-start_time,dt,time.perf_counter()-start_time-dt)


def register(plugin_manager):
    """ Register """
    plugin_manager.register_class(MouseHindlimb2DExtension)


def unregister(plugin_manager):
    """ Unregister """
    plugin_manager.unregister_class(MouseHindlimb2DExtension)
