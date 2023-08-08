from __future__ import annotations

import os
import socket
import struct
from enum import Enum

EXPECTED_SERVER_VERSION = 2


class CostFunction(Enum):
    """Enum for the different cost functions available in PyAutoscoper."""

    NORMALIZED_CROSS_CORRELATION = 0
    SUM_OF_ABSOLUTE_DIFFERENCES = 1


class OptimizationInitializationHeuristic(Enum):
    """Enum for the different optimization initialization heuristics available in PyAutoscoper."""

    CURRENT_FRAME = 0
    PREVIOUS_FRAME = 1
    LINEAR_EXTRAPOLATION = 2
    SPLINE_INTERPOLATION = 3


class OptimizationMethod(Enum):
    """Enum for the different optimization methods available in PyAutoscoper."""

    PARTICLE_SWARM_OPTIMIZATION = 0
    DOWNHILL_SIMPLEX = 1


class AutoscoperServerError(Exception):
    """Exception raised when the server reports an error."""

    def __str__(self) -> str:
        return f"Autoscoper Server error: {super().__str__()}"


class AutoscoperServerVersionMismatch(Exception):
    """Exception raised when the client attempt to connect to an unsupported server."""

    def __init__(self, server_version: int):
        self.server_version = server_version
        msg = f"server_version {self.server_version}, expected_version {EXPECTED_SERVER_VERSION}"
        super().__init__(msg)

    def __str__(self) -> str:
        return f"Autoscoper Server Version mismatch: {super().__str__()}"


class AutoscoperConnectionError(Exception):
    """Exception raised when the connection to the server is lost."""

    def __str__(self) -> str:
        return f"Error communicating with Autoscoper server: {super().__str__()}"


class AutoscoperConnection:
    def __init__(self, address: str = "127.0.0.1", verbose: bool = False) -> None:
        self.address = address
        self.verbose = verbose
        self.socket = self._openConnection()
        self._checkVersion()

    def __str__(self) -> str:
        return f"Autoscoper connection to {self.address}"

    def __repr__(self) -> str:
        return f"AutoscoperConnection('{self.address}', verbose={self.verbose})"

    def _wait_for_server(self) -> bytes:
        """
        Internal function, should not be called by a user.

        Waits for the server response after sending a message

        :return: The response from the server
        :rtype: bytes
        """
        while True:
            data = self.socket.recv(1024)
            if data:
                return data

    def _pack_data(self, *args: tuple) -> bytearray:
        """
        Internal function, should not be called by a user.

        Packs the given arguments into a bytearray.

        :param args: The arguments to pack
        :type args: tuple
        :return: The packed data
        :rtype: bytearray
        :raises Exception: If the argument type is invalid
        """
        packed_data = bytearray()
        for arg in args:
            if isinstance(arg, int):
                packed_data.extend(arg.to_bytes(4, byteorder="little", signed=False))
            elif isinstance(arg, float):
                packed_data.extend(struct.pack("d", arg))
            elif isinstance(arg, str):
                packed_data.extend(arg.encode("utf-8"))
            else:
                raise Exception(f"Invalid argument type: {type(arg)}")
        return packed_data

    def _send_command(self, command: int, *args: tuple) -> bytes:
        """
        Send a command to the server and wait for the response.

        :param command: The command byte to send
        :type command: int
        :param args: The arguments to send with the command
        :type args: tuple
        :return: The response from the server
        :rtype: bytes
        :raises Exception: If the server response is invalid
        """
        packed_data = self._pack_data(*args)
        b = bytearray()
        b.append(command)
        b.extend(packed_data)
        try:
            self.socket.sendall(b)
            response = self._wait_for_server()
        except OSError as e:
            raise AutoscoperConnectionError(e) from e
        if response[0] != command:
            self.closeConnection()
            raise AutoscoperServerError(f"received {response[0]}, expected {command}")
        return response

    def _test_connection(self) -> bool:
        """
        Test the connection to the PyAutoscoper server.

        :rtype: Boolean
        :raises Exception: If the connection is not successful
        """
        if self.verbose:
            print("Testing connection")
        self._send_command(0x00)
        return True

    def _openConnection(self) -> socket.socket:
        """
        Internal function, should not be called by a user.

        Open a tcp connection to the given address and port.

        Called automatically upon init.

        :return: The socket object
        :rtype: socket.socket
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.address, 30007))
        return s

    def _checkVersion(self):
        """
        Internal function, should not be called by a user.

        Checks that the server version is compatible with the client version.

        Called automatically upon init.

        :raises AutoscoperServerVersionMismatch: If the server version does not match the client version
        """
        response = self._send_command(0x10)  # 16
        server_version = struct.unpack("i", response[1:])[0]  # version string formatted as "NUMBER"

        if server_version != EXPECTED_SERVER_VERSION:
            raise AutoscoperServerVersionMismatch(server_version)

        if self.verbose:
            print(f"Server version: {server_version}")

    @property
    def is_connected(self) -> bool:
        """
        Returns the status of the connection to the PyAutoscoper server.

        :rtype: Boolean
        """
        try:
            self._test_connection()
            return True
        except (AutoscoperServerError, AutoscoperConnectionError):
            return False

    def loadTrial(self, trial_file: str):
        """
        Load a trial file into the PyAutoscoper server.

        :param trial_file: The path to the trial file to load
        :type trial_file: str
        :raises AutoscoperServerError: If the server fails to load the trial file
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        if self.verbose:
            print(f"Loading trial file: {trial_file}")
        if not os.path.exists(trial_file):
            raise AutoscoperServerError(f"File not found: {trial_file}")
        self._send_command(0x01, trial_file)

    def loadTrackingData(
        self,
        volume: int,
        tracking_data: str,
        is_matrix: bool = True,
        is_rows: bool = True,
        is_with_commas: bool = True,
        is_cm: bool = False,
        is_rad: bool = False,
        interpolate: bool = False,
    ):
        """
        Load tracking data into the PyAutoscoper server.

        :param volume: The volume to load the tracking data into
        :type volume: int
        :param tracking_data: The path to the tracking data to load
        :type tracking_data: str
        :param is_matrix: Optional - If true, the tracking data will be loaded as a 4 by 4 matrix.
          If false, the tracking data will be loaded in xyz roll pitch yaw format. Defaults to true.
        :type is_matrix: bool
        :param is_rows: Optional - If true, the tracking data will be loaded as rows.
          If false, the tracking data will be loaded as columns. Defaults to true.
        :type is_rows: bool
        :param is_with_commas: Optional - If true, the tracking data will be loaded with commas.
          If false, the tracking data will be loaded with spaces. Defaults to true.
        :type is_with_commas: bool
        :param is_cm: Optional - If true, the tracking data will be loaded in cm.
          If false, the tracking data will be loaded in mm. Defaults to false.
        :type is_cm: bool
        :param is_rad: Optional - If true, the tracking data will be loaded in radians.
          If false, the tracking data will be loaded in degrees. Defaults to false.
        :type is_rad: bool
        :param interpolate: Optional - If true, the tracking data will be interpolated using the spline method.
          If false, the tracking data will be saved as is (with NaN values). Defaults to false.
        :type interpolate: bool
        :raises AutoscoperServerError: If the server fails to load the tracking data
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        if self.verbose:
            print(f"Loading tracking data: {tracking_data}")
        if not os.path.exists(tracking_data):
            raise AutoscoperServerError(f"Tracking data not found: {tracking_data}")
        self._send_command(
            0x02,
            volume,
            int(is_matrix),
            int(is_rows),
            int(is_with_commas),
            int(is_cm),
            int(is_rad),
            int(interpolate),
            tracking_data,
        )

    def saveTracking(
        self,
        volume: int,
        tracking_file: str,
        save_as_matrix: bool = True,
        save_as_rows: bool = True,
        save_with_commas: bool = True,
        convert_to_cm: bool = False,
        convert_to_rad: bool = False,
        interpolate: bool = False,
    ):
        """
        Save tracking data from the PyAutoscoper server.

        :param volume: The volume to save the tracking data from
        :type volume: int
        :param tracking_file: The path to the tracking data to save
        :type tracking_file: str
        :param save_as_matrix: Optional - If true, the tracking data will be saved as a 4 by 4 matrix.
          If false, the tracking data will be saved in xyz roll pitch yaw format. Defaults to true.
        :type save_as_matrix: bool
        :param save_as_rows: Optional - If true, the tracking data will be saved as rows.
          If false, the tracking data will be saved as columns. Defaults to true.
        :type save_as_rows: bool
        :param save_with_commas: Optional - If true, the tracking data will be saved with commas.
          If false, the tracking data will be saved with spaces. Defaults to true.
        :type save_with_commas: bool
        :param convert_to_cm: Optional - If true, the tracking data will be converted to cm.
          If false, the tracking data will be saved in mm. Defaults to false.
        :type convert_to_cm: bool
        :param convert_to_rad: Optional - If true, the tracking data will be converted to radians.
          If false, the tracking data will be saved in degrees. Defaults to false.
        :type convert_to_rad: bool
        :param interpolate: Optional - If true, the tracking data will be interpolated using the spline method.
          If false, the tracking data will be saved as is (with NaN values). Defaults to false.
        :type interpolate: bool
        :raises AutoscoperServerError: If the server fails to save the tracking data
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        if self.verbose:
            print(f"Saving tracking data: {tracking_file}")

        self._send_command(
            0x03,
            volume,
            int(save_as_matrix),
            int(save_as_rows),
            int(save_with_commas),
            int(convert_to_cm),
            int(convert_to_rad),
            int(interpolate),
            tracking_file,
        )

    def loadFilters(self, camera: int, settings_file: str):
        """
        Load filter settings into the PyAutoscoper server.

        :param camera: The camera to load the filter settings into
        :type camera: int
        :param settings_file: The path to the filter settings to load
        :type settings_file: str
        :raises AutoscoperServerError: If the server fails to load the filter settings
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        if self.verbose:
            print(f"Loading filter settings: {settings_file}")
        if not os.path.exists(settings_file):
            raise AutoscoperServerError(f"Filter settings not found: {settings_file}")
        self._send_command(0x04, camera, settings_file)

    def setFrame(self, frame: int):
        """
        Set the frame to be used for the next acquisition.

        :param frame: The frame to be used for the next acquisition
        :type frame: int
        :raises AutoscoperServerError: If the server fails to set the frame
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        if self.verbose:
            print(f"Setting frame: {frame}")
        self._send_command(0x05, frame)

    def getPose(self, volume: int, frame: int) -> list[float]:
        """
        Get the pose of the volume at the specified frame.

        :param volume: The volume to get the pose of
        :type volume: int
        :param frame: The frame to get the pose at
        :type frame: int
        :return: The pose of the volume at the specified frame
        :rtype: list[float]
        :raises AutoscoperServerError: If the server fails to get the pose
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        if self.verbose:
            print(f"Getting pose for volume {volume} on frame {frame}")
        response = self._send_command(0x06, volume, frame)
        return [
            struct.unpack("d", response[1:9])[0],
            struct.unpack("d", response[9:17])[0],
            struct.unpack("d", response[17:25])[0],
            struct.unpack("d", response[25:33])[0],
            struct.unpack("d", response[33:41])[0],
            struct.unpack("d", response[41:49])[0],
        ]

    def setPose(self, volume: int, frame: int, pose: list[float]):
        """
        Set the pose of the volume at the specified frame.

        :param volume: The volume to set the pose of
        :type volume: int
        :param frame: The frame to set the pose at
        :type frame: int
        :param pose: The pose to set the volume to
        :type pose: list[float]
        :raises AutoscoperServerError: If the server fails to set the pose
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        if self.verbose:
            print(f"Setting pose {pose} for volume {volume} on frame {frame}")
        self._send_command(0x07, volume, frame, *pose)

    def getNCC(self, volume: int, pose: list[float]) -> list[float]:
        """
        Get the normalized cross correlation of the volume at the specified pose.

        :param volume: The volume to get the NCC of
        :type volume: int
        :param pose: The pose to get the NCC at
        :type pose: list[float]
        :return: The NCC of the volume at the specified pose
        :rtype: list[float]
        :raises AutoscoperServerError: If the server fails to get the NCC
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        if self.verbose:
            print(f"Getting NCC for volume {volume} on pose {pose}")
        response = self._send_command(0x08, volume, *pose)
        ncc = []
        for i in range(0, 2):
            val = response[2 + (i) * 8 : 10 + (i) * 8]
            ncc.append(struct.unpack("d", val)[0])
        return ncc

    def setBackground(self, threshold: float):
        """
        Set the background threshold.

        :param threshold: The background threshold
        :type threshold: float
        :raises AutoscoperServerError: If the server fails to set the background threshold
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        if self.verbose:
            print(f"Setting background threshold: {threshold}")
        self._send_command(0x09, threshold)

    def getImageCropped(self, volume: int, camera: int, pose: list[float]) -> list[float]:
        """
        Get the cropped image of the volume at the specified pose.

        :param volume: The volume to get the image of
        :type volume: int
        :param camera: The camera to get the image from
        :type camera: int
        :param pose: The pose to get the image at
        :type pose: list[float]
        :return: The cropped image of the volume at the specified pose
        :rtype: list[float]
        :raises AutoscoperServerError: If the server fails to get the image
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        if self.verbose:
            print(f"Getting image for volume {volume} on pose {pose} from camera {camera}")
        response = self._send_command(0x0A, volume, camera, *pose)  # 10
        width = struct.unpack("i", response[1:5])[0]
        height = struct.unpack("i", response[5:9])[0]
        img_data = response[9:]
        return [width, height, img_data]

    def optimizeFrame(
        self,
        volume: int,
        frame: int,
        repeats: int,
        max_itr: int,
        min_lim: float,
        max_lim: float,
        max_stall_itr: int,
        dframe: int,
        opt_method: OptimizationMethod,
        cf_model: CostFunction,
        opt_init_heuristic: OptimizationInitializationHeuristic,
    ):
        """
        Optimize the pose of the volume at the specified frame.

        :param volume: The volume to optimize
        :type volume: int
        :param frame: The frame to optimize
        :type frame: int
        :param repeats: The number of times to repeat the optimization
        :type repeats: int
        :param max_itr: The maximum number of iterations to run
        :type max_itr: int
        :param min_lim: The minimum limit of the optimization
        :type min_lim: float
        :param max_lim: The maximum limit of the optimization
        :type max_lim: float
        :param max_stall_itr: The maximum number of iterations to stall
        :type max_stall_itr: int
        :param dframe: The amount of frames to skip
        :type dframe: int
        :param opt_method: The optimization method to use.
        :type opt_method: int or :const:`~OptimizationMethod`
        :param cf_model: The cost function to use. :const:`~CostFunction.NORMALIZED_CROSS_CORRELATION`
          for Bone Models,  :const:`~CostFunction.SUM_OF_ABSOLUTE_DIFFERENCES` for Implant Models.
        :type cf_model: int or :const:`~CostFunction`
        :param opt_init_heuristic: The heuristic to initialize the optimization.
          See :const:`~OptimizationInitializationHeuristic`.
        :type opt_init_heuristic: int or :const:`~OptimizationInitializationHeuristic`

        :raises AutoscoperServerError: If the server fails to optimize the frame
        :raises AutoscoperConnectionError: If the connection to the server is lost
        :raises ValueError: If parameters accepting an enum value are incorrectly specified.

        .. versionadded:: 2

          The `opt_init_heuristic` parameter.
        """
        if not isinstance(opt_init_heuristic, OptimizationInitializationHeuristic):
            opt_init_heuristic = OptimizationInitializationHeuristic(opt_init_heuristic)

        if not isinstance(cf_model, CostFunction):
            cf_model = CostFunction(cf_model)

        if not isinstance(opt_method, OptimizationMethod):
            opt_method = OptimizationMethod(opt_method)

        if self.verbose:
            print(f"Optimizing volume {volume} on frame {frame}")
        self._send_command(
            0x0B,  # 11
            volume,
            frame,
            repeats,
            max_itr,
            float(min_lim),
            float(max_lim),
            max_stall_itr,
            dframe,
            opt_method.value,
            cf_model.value,
            opt_init_heuristic.value,
        )

    def saveFullDRR(self):
        """
        Save the full DRR.

        :raises AutoscoperServerError: If the server fails to save the full DRR
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        self._send_command(0x0C)  # 12

    def closeConnection(self):
        """
        Close the connection to the server.

        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        b = bytearray()
        # convert 13 to bytes
        b.append(0x0D)
        try:
            self.socket.sendall(b)
        except OSError as e:
            raise AutoscoperConnectionError("Connection to server lost") from e
        self.socket.close()

    def trackingDialog(
        self,
        volume: int,
        start_frame: int,
        end_frame: int,
        frame_skip: int = 1,
        repeats: int = 1,
        max_itr: int = 1000,
        min_lim: float = -3.0,
        max_lim: float = 3.0,
        max_stall_itr: int = 25,
        opt_method: OptimizationMethod = OptimizationMethod.PARTICLE_SWARM_OPTIMIZATION,
        cf_model: CostFunction = CostFunction.NORMALIZED_CROSS_CORRELATION,
        opt_init_heuristic: OptimizationInitializationHeuristic = OptimizationInitializationHeuristic.PREVIOUS_FRAME,
    ):
        """
        Automatically tracks the volume across the given frames.

        :param volume: The id of the volume to be tracked
        :type volume: int
        :param start_frame: The frame to start the tracking on
        :type start_frame: int
        :param end_frame: The frame to end the tracking on
        :type end_frame: int
        :param frame_skip: The amount of frames to skip over during tracking
        :type frame_skip: int
        :param repeats: The number of times to repeat the optimization
        :type repeats: int
        :param max_itr: The maximum number of iterations to run
        :type max_itr: int
        :param min_lim: The minimum limit of the optimization
        :type min_lim: float
        :param max_lim: The maximum limit of the optimization
        :type max_lim: float
        :param max_stall_itr: The maximum number of iterations to stall
        :type max_stall_itr: int
        :param opt_method: The optimization method to use.
        :type opt_method: int or :const:`~OptimizationMethod`
        :param cf_model: The cost function to use. :const:`~CostFunction.NORMALIZED_CROSS_CORRELATION`
          for Bone Models,  :const:`~CostFunction.SUM_OF_ABSOLUTE_DIFFERENCES` for Implant Models.
        :type cf_model: int or :const:`~CostFunction`
        :param opt_init_heuristic: The heuristic to initialize the optimization.
          See :const:`~OptimizationInitializationHeuristic`.
        :type opt_init_heuristic: int or :const:`~OptimizationInitializationHeuristic`

        :raises AutoscoperServerError: If the server fails to track the volume
        :raises AutoscoperConnectionError: If the connection to the server is lost
        :raises ValueError: If parameters accepting an enum value are incorrectly specified.

        .. versionadded:: 2

          The `opt_init_heuristic` parameter.
        """
        if self.verbose:
            print(f"Automated tracking of volume {volume} from frame {start_frame} to {end_frame}.\n")
        for frame in range(start_frame, end_frame):
            self.setFrame(frame=frame)
            self.optimizeFrame(
                volume=volume,
                frame=frame,
                repeats=repeats,
                max_itr=max_itr,
                min_lim=min_lim,
                max_lim=max_lim,
                max_stall_itr=max_stall_itr,
                dframe=frame_skip,
                opt_method=opt_method,
                cf_model=cf_model,
                opt_init_heuristic=opt_init_heuristic,
            )

    def getNumVolumes(self) -> int:
        """
        Get the number of volumes in the scene.

        :return: The number of volumes in the scene
        :rtype: int
        :raises AutoscoperServerError: If the server fails to get the number of volumes
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        response = self._send_command(0x0E)  # 14
        num_volume = struct.unpack("i", response[1:])[0]
        return num_volume

    def getNumFrames(self) -> int:
        """
        Get the number of frames in the scene.

        :return: The number of frames in the scene
        :rtype: int
        :raises AutoscoperServerError: If the server fails to get the number of frames
        :raises AutoscoperConnectionError: If the connection to the server is lost
        """
        response = self._send_command(0x0F)  # 15
        num_frames = struct.unpack("i", response[1:])[0]
        return num_frames
