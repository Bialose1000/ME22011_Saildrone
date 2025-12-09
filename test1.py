import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# -------------------------------------------------------
# 1.  Matplotlib canvas embedded in PyQt
# -------------------------------------------------------
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        super().__init__(self.fig)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.trail, = self.ax.plot([], [], 'b-', lw=2)
        self.dot,   = self.ax.plot([], [], 'ro', ms=6)


# -------------------------------------------------------
# 2.  Main GUI Window
# -------------------------------------------------------
class SaildroneGUI(QtWidgets.QWidget):
    def __init__(self, x_traj, y_traj, t_vals):
        super().__init__()

        self.x = x_traj
        self.y = y_traj
        self.t = t_vals
        self.index = 0
        self.running = False

        # Layout -------------------------------------------------------
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # Canvas
        self.canvas = MplCanvas()
        layout.addWidget(self.canvas)

        # Controls -----------------------------------------------------
        btn_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_layout)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.reset_btn = QtWidgets.QPushButton("Reset")

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.reset_btn)

        # Connect buttons
        self.start_btn.clicked.connect(self.start)
        self.pause_btn.clicked.connect(self.pause)
        self.reset_btn.clicked.connect(self.reset)

        # Readout panel ------------------------------------------------
        self.readout = QtWidgets.QLabel("t = 0.0 s")
        layout.addWidget(self.readout)

        # Timer for animation
        self.timer = QtCore.QTimer()
        self.timer.setInterval(40)  # 25 FPS
        self.timer.timeout.connect(self.update_plot)

        # Set axes limits
        self.canvas.ax.set_xlim(np.min(self.x)-5, np.max(self.x)+5)
        self.canvas.ax.set_ylim(np.min(self.y)-5, np.max(self.y)+5)

        self.setWindowTitle("Saildrone Simulation")
        self.resize(800, 800)

    # ---------------------------------------------------
    # Button actions
    # ---------------------------------------------------
    def start(self):
        self.running = True
        self.timer.start()

    def pause(self):
        self.running = False
        self.timer.stop()

    def reset(self):
        self.running = False
        self.timer.stop()
        self.index = 0
        self.update_plot()

    # ---------------------------------------------------
    # Animation update
    # ---------------------------------------------------
    def update_plot(self):
        if not self.running:
            return
        
        if self.index >= len(self.t):
            self.running = False
            return

        # Update trajectory and drone marker
        self.canvas.trail.set_data(self.x[:self.index], self.y[:self.index])
        self.canvas.dot.set_data(self.x[self.index], self.y[self.index])

        # Update numerical readout
        vx = np.gradient(self.x)[self.index]
        vy = np.gradient(self.y)[self.index]
        speed = np.sqrt(vx**2 + vy**2)

        heading = np.arctan2(vy, vx)

        self.readout.setText(
            f"t = {self.t[self.index]:.1f} s | "
            f"x = {self.x[self.index]:.2f} m | "
            f"y = {self.y[self.index]:.2f} m | "
            f"speed = {speed:.2f} m/s | "
            f"heading = {np.rad2deg(heading):.1f}°"
        )

        self.index += 1
        self.canvas.draw()


# -------------------------------------------------------
# 3.  Run the GUI (call this in your __main__ block)
# -------------------------------------------------------
def run_gui(x_traj, y_traj, t_vals):
    app = QtWidgets.QApplication(sys.argv)
    gui = SaildroneGUI(x_traj, y_traj, t_vals)
    gui.show()
    sys.exit(app.exec_())
