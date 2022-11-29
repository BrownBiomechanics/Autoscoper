import socket,struct

def wait_for_server(s):
    while True:
        data = s.recv(1024)
        if data:
            return data
def openConnection(address):
    """
    Open a tcp connection to the given address and port.

    :param address: The address to connect to
    :type address: str
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((address, 30007))
    return s

def loadTrial(s,trial_file):
    """
    Load a trial file into the PyAutoscoper server.

    :param s: The socket connection to the server
    :type s: socket.socket
    :param trial_file: The path to the trial file to load
    :type trial_file: str
    """
    b = bytearray()
    b.append(0x01)
    b.extend(trial_file.encode('utf-8'))
    s.sendall(b)
    wait_for_server(s)
    

def loadTrackingData(s,volume,tracking_data,save_as_matrix=True,save_as_rows=True,save_with_commas=True,convert_to_cm=False,convert_to_rad=False,interpolate=False):
    """
    Load tracking data into the PyAutoscoper server.

    :param s: The socket connection to the server
    :type s: socket.socket
    :param volume: The volume to load the tracking data into
    :type volume: int
    :param tracking_data: The path to the tracking data to load
    :type tracking_data: str
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
    """
    b = bytearray()
    b.append(0x02)
    b.extend(volume.to_bytes(4, byteorder='little', signed=False))
    b.extend((int(save_as_matrix)).to_bytes(4, byteorder='little', signed=False))
    b.extend((int(save_as_rows)).to_bytes(4, byteorder='little', signed=False))
    b.extend((int(save_with_commas)).to_bytes(4, byteorder='little', signed=False))
    b.extend((int(convert_to_cm)).to_bytes(4, byteorder='little', signed=False))
    b.extend((int(convert_to_rad)).to_bytes(4, byteorder='little', signed=False))
    b.extend((int(interpolate)).to_bytes(4, byteorder='little', signed=False))
    b.extend(tracking_data.encode('utf-8'))
    s.sendall(b)
    wait_for_server(s)

def saveTracking(s,volume,tracking_file,save_as_matrix=True,save_as_rows=True,save_with_commas=True,convert_to_cm=False,convert_to_rad=False,interpolate=False):
    """
    Save tracking data from the PyAutoscoper server.

    :param s: The socket connection to the server
    :type s: socket.socket
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
    """
    b = bytearray()
    b.append(0x03)
    b.extend(volume.to_bytes(4, byteorder='little', signed=False))
    b.extend((int(save_as_matrix)).to_bytes(4, byteorder='little', signed=False))
    b.extend((int(save_as_rows)).to_bytes(4, byteorder='little', signed=False))
    b.extend((int(save_with_commas)).to_bytes(4, byteorder='little', signed=False))
    b.extend((int(convert_to_cm)).to_bytes(4, byteorder='little', signed=False))
    b.extend((int(convert_to_rad)).to_bytes(4, byteorder='little', signed=False))
    b.extend((int(interpolate)).to_bytes(4, byteorder='little', signed=False))
    b.extend(tracking_file.encode('utf-8'))
    s.sendall(b)
    wait_for_server(s)

def loadFilters(s,camera,settings_file):
    """
    Load filter settings into the PyAutoscoper server.

    :param s: The socket connection to the server
    :type s: socket.socket
    :param camera: The camera to load the filter settings into
    :type camera: int
    :param settings_file: The path to the filter settings to load
    :type settings_file: str
    """
    b = bytearray()
    b.append(0x04)
    b.extend(camera.to_bytes(4, byteorder='little', signed=False))
    b.extend(settings_file.encode('utf-8'))
    s.sendall(b)
    wait_for_server(s)

def setFrame(s,frame):
    """
    Set the frame to be used for the next acquisition.

    :param s: The socket connection to the server
    :type s: socket.socket
    :param frame: The frame to be used for the next acquisition
    :type frame: int
    """
    b = bytearray()
    b.append(0x05)
    b.extend(frame.to_bytes(4, byteorder='little', signed=False))
    s.sendall(b)
    wait_for_server(s)

def getPose(s,volume,frame):
    """
    Get the pose of the volume at the specified frame.

    :param s: The socket connection to the server
    :type s: socket.socket
    :param volume: The volume to get the pose of
    :type volume: int
    :param frame: The frame to get the pose at
    :type frame: int
    :return: The pose of the volume at the specified frame
    :rtype: list[float]
    """
    b = bytearray()
    b.append(0x06)
    b.extend(volume.to_bytes(4, byteorder='little', signed=False))
    b.extend(frame.to_bytes(4, byteorder='little', signed=False))
    s.sendall(b)
    data = wait_for_server(s)
    data = bytearray(data)
    return [struct.unpack('d', data[1:9])[0], struct.unpack('d', data[9:17])[0], struct.unpack('d', data[17:25])[0], struct.unpack('d', data[25:33])[0], struct.unpack('d', data[33:41])[0], struct.unpack('d', data[41:49])[0]]

def setPose(s,volume,frame,pose):
    """
    Set the pose of the volume at the specified frame.

    :param s: The socket connection to the server
    :type s: socket.socket
    :param volume: The volume to set the pose of
    :type volume: int
    :param frame: The frame to set the pose at
    :type frame: int
    :param pose: The pose to set the volume to
    :type pose: list[float]
    """
    b = bytearray()
    b.append(0x07)
    b.extend(volume.to_bytes(4, byteorder='little', signed=False))
    b.extend(frame.to_bytes(4, byteorder='little', signed=False))
    b.extend(struct.pack('d', pose[0]))
    b.extend(struct.pack('d', pose[1]))
    b.extend(struct.pack('d', pose[2]))
    b.extend(struct.pack('d', pose[3]))
    b.extend(struct.pack('d', pose[4]))
    b.extend(struct.pack('d', pose[5]))
    s.sendall(b)
    wait_for_server(s)

def getNCC(s,volume,pose):
    """
    Get the normalized cross correlation of the volume at the specified pose.

    :param s: The socket connection to the server
    :type s: socket.socket
    :param volume: The volume to get the NCC of
    :type volume: int
    :param pose: The pose to get the NCC at
    :type pose: list[float]
    :return: The NCC of the volume at the specified pose
    :rtype: float
    """
    b = bytearray()
    b.append(0x08)
    b.extend(volume.to_bytes(4, byteorder='little', signed=False))
    b.extend(struct.pack('d', pose[0]))
    b.extend(struct.pack('d', pose[1]))
    b.extend(struct.pack('d', pose[2]))
    b.extend(struct.pack('d', pose[3]))
    b.extend(struct.pack('d', pose[4]))
    b.extend(struct.pack('d', pose[5]))
    s.sendall(b)
    data = wait_for_server(s)
    data = bytearray(data)
    ncc = []
    for i in range(0, 2):
        val = data[2 + (i)*8: 10+(i)*8]
        ncc.append(struct.unpack('d', val)[0])
    return ncc

def setBackground(s,threshold):
    """
    Set the background threshold.

    :param s: The socket connection to the server
    :type s: socket.socket
    :param threshold: The background threshold
    :type threshold: float
    """
    b = bytearray()
    b.append(0x09)
    b.extend(struct.pack('d', threshold))
    s.sendall(b)
    wait_for_server(s)

def getImageCropped(s,volume,camera,pose):
    """
    Get the cropped image of the volume at the specified pose.

    :param s: The socket connection to the server
    :type s: socket.socket
    :param volume: The volume to get the image of
    :type volume: int
    :param camera: The camera to get the image from
    :type camera: int
    :param pose: The pose to get the image at
    :type pose: list[float]
    :return: The cropped image of the volume at the specified pose
    :rtype: list[float]
    """
    b = bytearray()
    b.append(0x0A)
    b.extend(volume.to_bytes(4, byteorder='little', signed=False))
    b.extend(camera.to_bytes(4, byteorder='little', signed=False))
    b.extend(struct.pack('d', pose[0]))
    b.extend(struct.pack('d', pose[1]))
    b.extend(struct.pack('d', pose[2]))
    b.extend(struct.pack('d', pose[3]))
    b.extend(struct.pack('d', pose[4]))
    b.extend(struct.pack('d', pose[5]))
    s.sendall(b)
    data = wait_for_server(s)
    data = bytearray(data)
    width = struct.unpack('i', data[1:5])[0]
    height = struct.unpack('i', data[5:9])[0]
    img_data = data[9:]
    return [width, height, img_data]

def optimizeFrame(s,volume,frame,repeats,max_itr,min_lim,max_lim,max_stall_itr):
    """
    Optimize the pose of the volume at the specified frame.

    :param s: The socket connection to the server
    :type s: socket.socket
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
    """
    b = bytearray()
    b.append(0x0B)
    b.extend(volume.to_bytes(4, byteorder='little', signed=False))
    b.extend(frame.to_bytes(4, byteorder='little', signed=False))
    b.extend(repeats.to_bytes(4, byteorder='little', signed=False))
    b.extend(max_itr.to_bytes(4, byteorder='little', signed=False))
    b.extend(struct.pack('d', min_lim))
    b.extend(struct.pack('d', max_lim))
    b.extend(max_stall_itr.to_bytes(4, byteorder='little', signed=False))
    s.sendall(b)
    wait_for_server(s)

def saveFullDRR(s):
    """
    Save the full DRR.

    :param s: The socket connection to the server
    :type s: socket.socket
    """
    b = bytearray()
    b.append(0x0C)
    s.sendall(b)
    wait_for_server(s)

def closeConnection(s):
    """
    Close the connection to the server.

    :param s: The socket connection to the server
    :type s: socket.socket
    """
    b = bytearray()
    b.append(0xFF)
    s.sendall(b)
    wait_for_server(s)    