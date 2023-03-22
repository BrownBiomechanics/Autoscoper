import socket, struct, os


class AutoscoperConnection:
    def __init__(self, address="127.0.0.1", verbose=False) -> None:
        self.address = address
        self.verbose = verbose
        self.socket = self._openConnection()
        self.is_connected = self.test_connection()

    def __str__(self):
        return f"Autoscoper connection to {self.address}"

    def __repr__(self):
        return f"AutoscoperConnection('{self.address}', verbose={self.verbose})"

    def _wait_for_server(self):
        """
        Internal function, should not be called by a user.

        Waits for the server response after sending a message
        """
        while True:
            data = self.socket.recv(1024)
            if data:
                return data

    def test_connection(self):
        """
        Test the connection to the PyAutoscoper server.

        :rtype: Boolean
        :raises Exception: If the connection is not successful
        """
        if self.verbose:
            print("Testing connection")
        b = bytearray()
        b.append(0x00)
        self.socket.sendall(b)
        response = self._wait_for_server()
        if int.from_bytes(response, byteorder="little", signed=False) != 0x00:
            self.closeConnection()
            raise Exception("Server Error testing connection")
        return True

    def _openConnection(self):
        """
        Internal function, should not be called by a user.

        Open a tcp connection to the given address and port.

        Called automatically upon init.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.address, 30007))
        return s

    def loadTrial(self, trial_file):
        """
        Load a trial file into the PyAutoscoper server.

        :param trial_file: The path to the trial file to load
        :type trial_file: str
        :raises Exception: If the trial file is not found, or If the server fails to load the trial file
        """
        if self.verbose:
            print(f"Loading trial file: {trial_file}")
        if not os.path.exists(trial_file):
            self.closeConnection()
            raise Exception("Trial file not found")
        b = bytearray()
        b.append(0x01)
        b.extend(trial_file.encode("utf-8"))
        self.socket.sendall(b)
        response = self._wait_for_server()
        if int.from_bytes(response, byteorder="little", signed=False) != 0x01:
            self.closeConnection()
            raise Exception("Server Error loading trial file")

    def loadTrackingData(
        self,
        volume,
        tracking_data,
        is_matrix=True,
        is_rows=True,
        is_with_commas=True,
        is_cm=False,
        is_rad=False,
        interpolate=False,
    ):
        """
        Load tracking data into the PyAutoscoper server.

        :param volume: The volume to load the tracking data into
        :type volume: int
        :param tracking_data: The path to the tracking data to load
        :type tracking_data: str
        :param is_matrix: Optional - If true, the tracking data will be loaded as a 4 by 4 matrix. If false, the tracking data will be loaded in xyz roll pitch yaw format. Defaults to true.
        :type is_matrix: bool
        :param is_rows: Optional - If true, the tracking data will be loaded as rows. If false, the tracking data will be loaded as columns. Defaults to true.
        :type is_rows: bool
        :param is_with_commas: Optional - If true, the tracking data will be loaded with commas. If false, the tracking data will be loaded with spaces. Defaults to true.
        :type is_with_commas: bool
        :param is_cm: Optional - If true, the tracking data will be loaded in cm. If false, the tracking data will be loaded in mm. Defaults to false.
        :type is_cm: bool
        :param is_rad: Optional - If true, the tracking data will be loaded in radians. If false, the tracking data will be loaded in degrees. Defaults to false.
        :type is_rad: bool
        :param interpolate: Optional - If true, the tracking data will be interpolated using the spline method. If false, the tracking data will be saved as is (with NaN values). Defaults to false.
        :type interpolate: bool
        :raises Exception: If the tracking data file is not found, or If the server fails to load the tracking data
        """
        if self.verbose:
            print(f"Loading tracking data: {tracking_data}")
        if not os.path.exists(tracking_data):
            self.closeConnection()
            raise Exception("Tracking data file not found")
        b = bytearray()
        b.append(0x02)
        b.extend(volume.to_bytes(4, byteorder="little", signed=False))
        b.extend((int(is_matrix)).to_bytes(4, byteorder="little", signed=False))
        b.extend((int(is_rows)).to_bytes(4, byteorder="little", signed=False))
        b.extend((int(is_with_commas)).to_bytes(4, byteorder="little", signed=False))
        b.extend((int(is_cm)).to_bytes(4, byteorder="little", signed=False))
        b.extend((int(is_rad)).to_bytes(4, byteorder="little", signed=False))
        b.extend((int(interpolate)).to_bytes(4, byteorder="little", signed=False))
        b.extend(tracking_data.encode("utf-8"))
        self.socket.sendall(b)
        response = self._wait_for_server()
        if int.from_bytes(response, byteorder="little", signed=False) != 0x02:
            self.closeConnection()
            raise Exception("Server Error loading tracking data")

    def saveTracking(
        self,
        volume,
        tracking_file,
        save_as_matrix=True,
        save_as_rows=True,
        save_with_commas=True,
        convert_to_cm=False,
        convert_to_rad=False,
        interpolate=False,
    ):
        """
        Save tracking data from the PyAutoscoper server.

        :param volume: The volume to save the tracking data from
        :type volume: int
        :param tracking_file: The path to the tracking data to save
        :type tracking_file: str
        :param save_as_matrix: Optional - If true, the tracking data will be saved as a 4 by 4 matrix. If false, the tracking data will be saved in xyz roll pitch yaw format. Defaults to true.
        :type save_as_matrix: bool
        :param save_as_rows: Optional - If true, the tracking data will be saved as rows. If false, the tracking data will be saved as columns. Defaults to true.
        :type save_as_rows: bool
        :param save_with_commas: Optional - If true, the tracking data will be saved with commas. If false, the tracking data will be saved with spaces. Defaults to true.
        :type save_with_commas: bool
        :param convert_to_cm: Optional - If true, the tracking data will be converted to cm. If false, the tracking data will be saved in mm. Defaults to false.
        :type convert_to_cm: bool
        :param convert_to_rad: Optional - If true, the tracking data will be converted to radians. If false, the tracking data will be saved in degrees. Defaults to false.
        :type convert_to_rad: bool
        :param interpolate: Optional - If true, the tracking data will be interpolated using the spline method. If false, the tracking data will be saved as is (with NaN values). Defaults to false.
        :type interpolate: bool
        :raises Exception: If the server fails to save the tracking data
        """
        if self.verbose:
            print(f"Saving tracking data: {tracking_file}")
        b = bytearray()
        b.append(0x03)
        b.extend(volume.to_bytes(4, byteorder="little", signed=False))
        b.extend((int(save_as_matrix)).to_bytes(4, byteorder="little", signed=False))
        b.extend((int(save_as_rows)).to_bytes(4, byteorder="little", signed=False))
        b.extend((int(save_with_commas)).to_bytes(4, byteorder="little", signed=False))
        b.extend((int(convert_to_cm)).to_bytes(4, byteorder="little", signed=False))
        b.extend((int(convert_to_rad)).to_bytes(4, byteorder="little", signed=False))
        b.extend((int(interpolate)).to_bytes(4, byteorder="little", signed=False))
        b.extend(tracking_file.encode("utf-8"))
        self.socket.sendall(b)
        response = self._wait_for_server()
        if int.from_bytes(response, byteorder="little", signed=False) != 0x03:
            self.closeConnection()
            raise Exception("Server Error saving tracking data")

    def loadFilters(self, camera, settings_file):
        """
        Load filter settings into the PyAutoscoper server.

        :param camera: The camera to load the filter settings into
        :type camera: int
        :param settings_file: The path to the filter settings to load
        :type settings_file: str
        :raises Exception: If the filter settings file is not found, or If the server fails to load the filter settings
        """
        if self.verbose:
            print(f"Loading filter settings: {settings_file}")
        if not os.path.exists(settings_file):
            self.closeConnection()
            raise Exception("Filter settings file not found")
        b = bytearray()
        b.append(0x04)
        b.extend(camera.to_bytes(4, byteorder="little", signed=False))
        b.extend(settings_file.encode("utf-8"))
        self.socket.sendall(b)
        response = self._wait_for_server()
        if int.from_bytes(response, byteorder="little", signed=False) != 0x04:
            self.closeConnection()
            raise Exception("Server Error loading filter settings")

    def setFrame(self, frame):
        """
        Set the frame to be used for the next acquisition.

        :param frame: The frame to be used for the next acquisition
        :type frame: int
        :raises Exception: If the server fails to set the frame
        """
        if self.verbose:
            print(f"Setting frame: {frame}")
        b = bytearray()
        b.append(0x05)
        b.extend(frame.to_bytes(4, byteorder="little", signed=False))
        self.socket.sendall(b)
        response = self._wait_for_server()
        if int.from_bytes(response, byteorder="little", signed=False) != 0x05:
            self.closeConnection()
            raise Exception("Server Error setting frame")

    def getPose(self, volume, frame):
        """
        Get the pose of the volume at the specified frame.

        :param volume: The volume to get the pose of
        :type volume: int
        :param frame: The frame to get the pose at
        :type frame: int
        :return: The pose of the volume at the specified frame
        :rtype: list[float]
        :raises Exception: If the server fails to get the pose
        """
        if self.verbose:
            print(f"Getting pose for volume {volume} on frame {frame}")
        b = bytearray()
        b.append(0x06)
        b.extend(volume.to_bytes(4, byteorder="little", signed=False))
        b.extend(frame.to_bytes(4, byteorder="little", signed=False))
        self.socket.sendall(b)
        response = self._wait_for_server()
        if response[0] != 0x06:
            self.closeConnection()
            raise Exception("Server Error getting pose")
        data = bytearray(response)
        return [
            struct.unpack("d", data[1:9])[0],
            struct.unpack("d", data[9:17])[0],
            struct.unpack("d", data[17:25])[0],
            struct.unpack("d", data[25:33])[0],
            struct.unpack("d", data[33:41])[0],
            struct.unpack("d", data[41:49])[0],
        ]

    def setPose(self, volume, frame, pose):
        """
        Set the pose of the volume at the specified frame.

        :param volume: The volume to set the pose of
        :type volume: int
        :param frame: The frame to set the pose at
        :type frame: int
        :param pose: The pose to set the volume to
        :type pose: list[float]
        :raises Exception: If the server fails to set the pose
        """
        if self.verbose:
            print(f"Setting pose {pose} for volume {volume} on frame {frame}")
        b = bytearray()
        b.append(0x07)
        b.extend(volume.to_bytes(4, byteorder="little", signed=False))
        b.extend(frame.to_bytes(4, byteorder="little", signed=False))
        b.extend(struct.pack("d", pose[0]))
        b.extend(struct.pack("d", pose[1]))
        b.extend(struct.pack("d", pose[2]))
        b.extend(struct.pack("d", pose[3]))
        b.extend(struct.pack("d", pose[4]))
        b.extend(struct.pack("d", pose[5]))
        self.socket.sendall(b)
        response = self._wait_for_server()
        if int.from_bytes(response, byteorder="little", signed=False) != 0x07:
            self.closeConnection()
            raise Exception("Server Error setting pose")

    def getNCC(self, volume, pose):
        """
        Get the normalized cross correlation of the volume at the specified pose.

        :param volume: The volume to get the NCC of
        :type volume: int
        :param pose: The pose to get the NCC at
        :type pose: list[float]
        :return: The NCC of the volume at the specified pose
        :rtype: list[float]
        :raises Exception: If the server fails to get the NCC
        """
        if self.verbose:
            print(f"Getting NCC for volume {volume} on pose {pose}")
        b = bytearray()
        b.append(0x08)
        b.extend(volume.to_bytes(4, byteorder="little", signed=False))
        b.extend(struct.pack("d", pose[0]))
        b.extend(struct.pack("d", pose[1]))
        b.extend(struct.pack("d", pose[2]))
        b.extend(struct.pack("d", pose[3]))
        b.extend(struct.pack("d", pose[4]))
        b.extend(struct.pack("d", pose[5]))
        self.socket.sendall(b)
        response = self._wait_for_server()
        if response[0] != 0x08:
            self.closeConnection()
            raise Exception("Server Error getting NCC")
        data = bytearray(response)
        ncc = []
        for i in range(0, 2):
            val = data[2 + (i) * 8 : 10 + (i) * 8]
            ncc.append(struct.unpack("d", val)[0])
        return ncc

    def setBackground(self, threshold):
        """
        Set the background threshold.

        :param threshold: The background threshold
        :type threshold: float
        :raises Exception: If the server fails to set the background threshold
        """
        if self.verbose:
            print(f"Setting background threshold: {threshold}")
        b = bytearray()
        b.append(0x09)
        b.extend(struct.pack("d", threshold))
        self.socket.sendall(b)
        response = self._wait_for_server()
        if int.from_bytes(response, byteorder="little", signed=False) != 0x09:
            self.closeConnection()
            raise Exception("Server Error setting background threshold")

    def getImageCropped(self, volume, camera, pose):
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
        :raises Exception: If the server fails to get the image
        """
        if self.verbose:
            print(
                f"Getting image for volume {volume} on pose {pose} from camera {camera}"
            )
        b = bytearray()
        b.append(0x0A)
        b.extend(volume.to_bytes(4, byteorder="little", signed=False))
        b.extend(camera.to_bytes(4, byteorder="little", signed=False))
        b.extend(struct.pack("d", pose[0]))
        b.extend(struct.pack("d", pose[1]))
        b.extend(struct.pack("d", pose[2]))
        b.extend(struct.pack("d", pose[3]))
        b.extend(struct.pack("d", pose[4]))
        b.extend(struct.pack("d", pose[5]))
        self.socket.sendall(b)
        response = self._wait_for_server()
        if response[0] != 0x0A:
            self.closeConnection()
            raise Exception("Server Error getting image")
        data = bytearray(response)
        width = struct.unpack("i", data[1:5])[0]
        height = struct.unpack("i", data[5:9])[0]
        img_data = data[9:]
        return [width, height, img_data]

    def optimizeFrame(
        self,
        volume,
        frame,
        repeats,
        max_itr,
        min_lim,
        max_lim,
        max_stall_itr,
        dframe,
        opt_method,
        cf_model,
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
        :param opt_method: The optimization method to use, 0 for Particle Swarm, 1 for Downhill Simplex
        :type opt_method: int
        :param cf_model: The cost function model to use, 0 for NCC (Bone Models), 1 for Sum of Absolute Differences (Implant Models)
        :type cf_model: int
        :raises Exception: If the server fails to optimize the frame
        """
        if opt_method not in [0, 1]:
            raise Exception("Invalid optimization method")
        if cf_model not in [0, 1]:
            raise Exception("Invalid cost function model")
        if self.verbose:
            print(f"Optimizing volume {volume} on frame {frame}")
        b = bytearray()
        b.append(0x0B)
        b.extend(volume.to_bytes(4, byteorder="little", signed=False))
        b.extend(frame.to_bytes(4, byteorder="little", signed=False))
        b.extend(repeats.to_bytes(4, byteorder="little", signed=False))
        b.extend(max_itr.to_bytes(4, byteorder="little", signed=False))
        b.extend(struct.pack("d", min_lim))
        b.extend(struct.pack("d", max_lim))
        b.extend(max_stall_itr.to_bytes(4, byteorder="little", signed=False))
        b.extend(dframe.to_bytes(4, byteorder="little", signed=False))
        b.extend(opt_method.to_bytes(4, byteorder="little", signed=False))
        b.extend(cf_model.to_bytes(4, byteorder="little", signed=False))
        self.socket.sendall(b)
        response = self._wait_for_server()
        if int.from_bytes(response, byteorder="little", signed=False) != 0x0B:
            self.closeConnection()
            raise Exception("Server Error optimizing frame")

    def saveFullDRR(self):
        """
        Save the full DRR.

        :raises Exception: If the server fails to save the full DRR
        """
        b = bytearray()
        b.append(0x0C)
        self.socket.sendall(b)
        response = self._wait_for_server()
        if int.from_bytes(response, byteorder="little", signed=False) != 0x0C:
            self.closeConnection()
            raise Exception("Server Error saving full DRR")

    def closeConnection(self):
        """
        Close the connection to the server.

        """
        b = bytearray()
        # convert 13 to bytes
        b.append(0x0D)
        self.socket.sendall(b)
        self.socket.close()
        self.is_connected = False

    def trackingDialog(
        self,
        volume,
        start_frame,
        end_frame,
        frame_skip=1,
        repeats=1,
        max_itr=1000,
        min_lim=-3,
        max_lim=3,
        max_stall_itr=25,
        opt_method=0,
        cf_model=0,
    ):
        """
        Automatically tracks the volume accross the given frames.

        Currently using previous frame for intial guess.

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
        """
        if self.verbose:
            print(
                f"Automated tracking of volume {volume} from frame {start_frame} to {end_frame}.\n"
            )
        for frame in range(start_frame, end_frame):
            self.setFrame(frame=frame)
            if frame != 0:
                pose = self.getPose(volume=volume, frame=(frame - 1))
                self.setPose(volume=volume, frame=frame, pose=pose)
            self.optimizeFrame(
                volume=volume,
                frame=frame,
                repeats=repeats,
                max_itr=max_itr,
                min_lim=min_lim,
                max_lim=max_lim,
                max_stall_itr=max_stall_itr,
                opt_method=opt_method,
                cf_model=cf_model,
                dframe=frame_skip,
            )

    def getNumVolumes(self):
        """
        Get the number of volumes in the scene.

        :return: The number of volumes in the scene
        :rtype: int
        :raises Exception: If the server fails to get the number of volumes
        """
        b = bytearray()
        # convert 14 to bytes
        b.append(0x0E)
        self.socket.sendall(b)
        response = self._wait_for_server()
        if response[0] != 0x0E:
            self.closeConnection()
            raise Exception("Server Error getting number of volumes")
        return int.from_bytes(response[1:], byteorder="little", signed=False)

    def getNumFrames(self):
        """
        Get the number of frames in the scene.

        :return: The number of frames in the scene
        :rtype: int
        :raises Exception: If the server fails to get the number of frames
        """
        b = bytearray()
        # convert 15 to bytes
        b.append(0x0F)
        self.socket.sendall(b)
        response = self._wait_for_server()
        if response[0] != 0x0F:
            self.closeConnection()
            raise Exception("Server Error getting number of frames")
        return int.from_bytes(response[1:], byteorder="little", signed=False)
