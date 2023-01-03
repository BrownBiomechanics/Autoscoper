classdef AutoscoperConnection
    properties (Access = private)
        address = "127.0.0.1";
        port = 30007;
        socket_descriptor;
    end
    methods
        function obj = AutoscoperConnection(address)
            % Creates a connection to the Autoscoper server
            obj.socket_descriptor = tcpclient(obj.address, obj.port);
            fopen(obj.socket_descriptor);
            if nargin == 1
                obj.address = address;
            end
        end

        function closeConnection(obj)
            % Closes the connection
            fwrite(autoscoper_socket,[13]);
            while autoscoper_socket.BytesAvailable == 0
                pause(1)
            end
            data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);
        end

        function loadTrial(obj, trial)
            % Loads a trial
            fwrite(obj.socket_descriptor, [1 trial]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function loadTrackingData(obj,volume,tracking_data,save_as_matrix,save_as_rows,save_with_commas,convert_to_cm,convert_to_rad,interpolate)
            % Loads a tracking data file for the given volume
            % only obj, volume, tracking_data are required
            % save_as_matrix, save_as_rows, save_with_commas, convert_to_cm, convert_to_rad, interpolate are optional

            % volume: the volume number to load
            % tracking_data: the tracking data to load
            % save_as_matrix: 1 to save as a matrix, 0 to save as xyzypr format
            % save_as_rows: 1 to save as rows, 0 to save as columns
            % save_with_commas: 1 to save with commas, 0 to save with spaces
            % convert_to_cm: 1 to convert to cm, 0 to leave in mm
            % convert_to_rad: 1 to convert to radians, 0 to leave in degrees
            % interpolate: 1 to interpolate(Spline), 0 to leave as is

            if nargin < 3
                error('Not enough input arguments')
            end
            if nargin < 4
                save_as_matrix = 1;
            end
            if nargin < 5
                save_as_rows = 1;
            end
            if nargin < 6
                save_with_commas = 1;
            end
            if nargin < 7
                convert_to_cm = 0;
            end
            if nargin < 8
                convert_to_rad = 0;
            end
            if nargin < 9
                interpolate = 0;
            end
            fwrite(obj.socket_descriptor, [2 typecast(int32(volume),'uint8') typecast(int32(save_as_matrix),'uint8') typecast(int32(save_as_rows),'uint8') typecast(int32(save_with_commas),'uint8') typecast(int32(convert_to_cm),'uint8') typecast(int32(convert_to_rad),'uint8') typecast(int32(interpolate),'uint8') tracking_data]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function saveTrackingData(obj,volume,tracking_data,save_as_matrix,save_as_rows,save_with_commas,convert_to_cm,convert_to_rad,interpolate)
            % Saves a tracking data file for the given volume
            % only obj, volume, tracking_data are required
            % save_as_matrix, save_as_rows, save_with_commas, convert_to_cm, convert_to_rad, interpolate are optional

            % volume: the volume number to save
            % tracking_data: the tracking data to save
            % save_as_matrix: 1 to save as a matrix, 0 to save as xyzypr format
            % save_as_rows: 1 to save as rows, 0 to save as columns
            % save_with_commas: 1 to save with commas, 0 to save with spaces
            % convert_to_cm: 1 to convert to cm, 0 to leave in mm
            % convert_to_rad: 1 to convert to radians, 0 to leave in degrees
            % interpolate: 1 to interpolate(Spline), 0 to leave as is

            if nargin < 3
                error('Not enough input arguments')
            end
            if nargin < 4
                save_as_matrix = 1;
            end
            if nargin < 5
                save_as_rows = 1;
            end
            if nargin < 6
                save_with_commas = 1;
            end
            if nargin < 7
                convert_to_cm = 0;
            end
            if nargin < 8
                convert_to_rad = 0;
            end
            if nargin < 9
                interpolate = 0;
            end
            fwrite(obj.socket_descriptor, [3 typecast(int32(volume),'uint8') typecast(int32(save_as_matrix),'uint8') typecast(int32(save_as_rows),'uint8') typecast(int32(save_with_commas),'uint8') typecast(int32(convert_to_cm),'uint8') typecast(int32(convert_to_rad),'uint8') typecast(int32(interpolate),'uint8') tracking_data]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function loadFilters(obj, camera, filter_file)
            % Loads a filter file for the given camera

            % camera: the camera number to load
            % filter_file: the filter file to load

            if nargin < 3
                error('Not enough input arguments')
            end
            fwrite(obj.socket_descriptor, [4 typecast(int32(camera),'uint8') filter_file]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function setFrame(obj,frame)
            % Sets the frame number

            % frame: the frame number to set

            if nargin < 2
                error('Not enough input arguments')
            end
            fwrite(obj.socket_descriptor, [5 typecast(int32(frame),'uint8')]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function pose = getPose(obj,volume,frame)
            % Gets the pose for the given volume and frame

            % volume: the volume number to get the pose for
            % frame: the frame number to get the pose for

            if nargin < 3
                error('Not enough input arguments')
            end
            fwrite(obj.socket_descriptor, [6 typecast(int32(volume),'uint8') typecast(int32(frame),'uint8')]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
            pose = [...
                typecast(uint8(data(2:9)),'double') ...
                typecast(uint8(data(10:17)),'double') ...
                typecast(uint8(data(18:25)),'double') ...
                typecast(uint8(data(26:33)),'double') ...
                typecast(uint8(data(34:41)),'double') ...
                typecast(uint8(data(42:49)),'double') ];
        end

        function setPose(obj,volume,frame,pose)
            % Sets the pose for the given volume and frame

            % volume: the volume number to set the pose for
            % frame: the frame number to set the pose for
            % pose: the pose to set

            if nargin < 4
                error('Not enough input arguments')
            end
            fwrite(obj.socket_descriptor, [7 typecast(int32(volume),'uint8') typecast(int32(frame),'uint8') typecast(double(pose),'uint8')]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function ncc = getNCC(obj, volume, pose)
            % Gets the NCC for the given volume and pose

            % volume: the volume number to get the NCC for
            % pose: the pose to get the NCC for

            if nargin < 3
                error('Not enough input arguments')
            end
            fwrite(obj.socket_descriptor, [8 typecast(int32(volume),'uint8') typecast(double(pose),'uint8')]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
            ncc = {};
            for i = 1:data(2)
                val = typecast(uint8(data(3 + (i-1)*8: 10+(i-1)*8)),'double')
                if(val == -99999)
                    val = NaN;
                end
                ncc = [ncc val];
            end
        end

        function setBackground(obj, threshold)
            % Sets the background threshold

            % threshold: the threshold to set

            if nargin < 2
                error('Not enough input arguments')
            end
            fwrite(obj.socket_descriptor, [9 typecast(double(threshold),'uint8')]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function getImageCropped(obj, volume, camera, pose)
            % not yet implemented

            % volume: the volume number to get the image for
            % camera: the camera number to get the image for
            % pose: the pose to get the image for

            if nargin < 4
                error('Not enough input arguments')
            end

            fwrite(obj.socket_descriptor, [10 typecast(int32(volume),'uint8') typecast(int32(camera),'uint8') typecast(double(pose),'uint8')]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function optimizeFrame(obj, volume, frame, repeats,max_itr,min_lim,max_lim,max_stall_itr,dframe,opt_method,cf_model)
            % Optimizes the given frame
            % only obj, volume, and frame are required
            % all other parameters are optional

            % volume: the volume number to optimize
            % frame: the frame number to optimize
            % repeats: the number of times to repeat the optimization
            % max_itr: the maximum number of iterations
            % min_lim: the minimum limit
            % max_lim: the maximum limit
            % max_stall_itr: the maximum number of iterations to stall
            % dframe: The amount of frames to skip
            % opt_method: The optimization method to use, 0 for Partical Swarm, 1 for Downhill Simplex
            % cf_model: The cost function model to use, 0 for NCC (Bone Models), 1 for Sum of Absolute Differences (Implant Models)

            if nargin < 3
                error('Not enough input arguments')
            end
            if nargin < 4
                repeats = 1;
            end
            if nargin < 5
                max_itr = 1000;
            end
            if nargin < 6
                min_lim = -3.0;
            end
            if nargin < 7
                max_lim = 3.0;
            end
            if nargin < 8
                max_stall_itr = 25;
            end
            if nargin < 9
                dframe = 1;
            end
            if nargin < 10
                opt_method = 0;
            end
            if nargin < 11
                cf_model = 0;
            end
            fwrite(obj.socket_descriptor, [11 typecast(int32(volume),'uint8') typecast(int32(frame),'uint8') typecast(int32(repeats),'uint8') typecast(int32(max_itr),'uint8') typecast(double(min_lim),'uint8') typecast(double(max_lim),'uint8') typecast(int32(max_stall_itr),'uint8') typecast(int32(dframe),'uint8') typecast(int32(opt_method),'uint8') typecast(int32(cf_model),'uint8')]);    
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function saveFullDRR(obj)
            % Saves the full DRR

            fwrite(obj.socket_descriptor, [12]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function trackingDialog(obj,startframe,endframe,repeats,max_itr,min_lim,max_lim,max_stall_itr,dframe,opt_method,cf_model)
            % Performs optimization on a range of frames
            % Only obj, startframe, and endframe are required
            % all other parameters are optional

            % startframe: the first frame to optimize
            % endframe: the last frame to optimize
            % repeats: the number of times to repeat the optimization
            % max_itr: the maximum number of iterations
            % min_lim: the minimum limit
            % max_lim: the maximum limit
            % max_stall_itr: the maximum number of iterations to stall
            % dframe: The amount of frames to skip
            % opt_method: The optimization method to use, 0 for Partical Swarm, 1 for Downhill Simplex
            % cf_model: The cost function model to use, 0 for NCC (Bone Models), 1 for Sum of Absolute Differences (Implant Models)

            if nargin < 3
                error('Not enough input arguments')
            end
            if nargin < 4
                repeats = 1;
            end
            if nargin < 5
                max_itr = 1000;
            end
            if nargin < 6
                min_lim = -3.0;
            end
            if nargin < 7
                max_lim = 3.0;
            end
            if nargin < 8
                max_stall_itr = 25;
            end
            if nargin < 9
                dframe = 1;
            end
            if nargin < 10
                opt_method = 0;
            end
            if nargin < 11
                cf_model = 0;
            end

            for i = startframe:endframe
                obj.setFrame(i);
                if i ~= 0
                    pose = obj.getPose(0,i-1);
                    obj.setPose(0,i,pose);
                end
                obj.optimizeFrame(0,i,repeats,max_itr,min_lim,max_lim,max_stall_itr,dframe,opt_method,cf_model);
            end
        end

        function ncc_out = getNCC_Sum(obj,volume,pose)
            % Gets the sum of the NCC for the given volume and pose

            % volume: the volume number to get the NCC for 
            % pose: the pose to get the NCC for

            if nargin < 3
                error('Not enough input arguments')
            end
            data = obj.getNCC(volume,pose);
            ncc = zeros(1,data(2));
            for i = 1:data(2)
                val = typecast(uint8(data(3 + (i-1)*8: 10+(i-1)*8)),'double') ;
                if(val < 0)
                    val = 1e3;
                end
                ncc(i) = val;
            end

            ncc_out  = ncc(1) + ncc(2);
        end

        function ncc_out = getNCC_This_Frame(obj,volume,frame)
            % Gets the NCC for the given volume and frame

            % volume: the volume number to get the NCC for 
            % frame: the frame to get the NCC for

            if nargin < 3
                error('Not enough input arguments')
            end
            pose = obj.getPose(volume,frame);
            data = obj.getNCC(volume,pose);
            ncc = zeros(1,data(2));
            for i = 1:data(2)
                val = typecast(uint8(data(3 + (i-1)*8: 10+(i-1)*8)),'double') ;
                if(val == -99999)
                    val = NaN;
                end
                ncc(i) = val;
            end

            ncc_out  = [ncc(1),ncc(2),ncc(1)*ncc(2)];
        end
    end
end