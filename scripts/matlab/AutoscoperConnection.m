classdef AutoscoperConnection
    properties (Access = private)
        address = "127.0.0.1";
        port = 30007;
        socket_descriptor;
    end
    methods
        function obj = AutoscoperConnection(address)
            % Creates a connection to the Autoscoper
            % supports single instance connection only

            v_old = isMATLABReleaseOlderThan("R2022a");
            if v_old
                obj.socket_descriptor = tcpip(obj.address, obj.port, 'NetworkRole', 'client');
            else
                obj.socket_descriptor = tcpclient(obj.address, obj.port);
            end

            fopen(obj.socket_descriptor);
            if nargin == 1
                obj.address = address;
            end
        end

        function closeConnection(obj)
            % Closes the connection

            % Since the server is explicitly asked to terminate the connection, no
            % reply from the server are expected.
            fwrite(obj.socket_descriptor,[13]);
        end

        function loadTrial(obj, path_to_cfg_file)
            % Loads a trial
            % path_to_cfg_file : full path to config (.cfg) file
            %   fail if incorrect filepath/ no file found

            fwrite(obj.socket_descriptor, [1 path_to_cfg_file]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function loadTrackingData(obj, volNum, tra_fileName, is_matrix, is_rows, is_csv, is_cm, is_rad, interpY)
            % Loads a tracking data file for the given volume
            % only obj, volume, tra_fileName are required
            % is_matrix, is_rows, is_with_commas, is_cm, is_rad, interpolate are optional

            % volN: the volume number to load
            % tra_fileName: the path to tracking data to load
            % is_matrix: 1 if the tracking data is a matrix, 0 if it is in xyzypr format
            % is_rows: 1 if the tracking data is in rows, 0 if it is in columns
            % is_csv: 1 if the tracking data is with commas, 0 if it is with spaces
            % is_cm: 1 if the tracking data is in cm, 0 if it is in mm
            % is_rad: 1 if the tracking data is in radians, 0 if it is in degrees
            % interpY: 1 to interpolate(Spline), 0 to leave as is

            if nargin < 3
                error('Not enough input arguments')
            end
            if nargin < 4
                is_matrix = 1;
            end
            if nargin < 5
                is_rows = 1;
            end
            if nargin < 6
                is_csv = 1;
            end
            if nargin < 7
                is_cm = 0;
            end
            if nargin < 8
                is_rad = 0;
            end
            if nargin < 9
                interpY = 0;
            end
            fwrite(obj.socket_descriptor, [2 typecast(int32(volNum),'uint8') typecast(int32(is_matrix),'uint8'),...
                typecast(int32(is_rows),'uint8') typecast(int32(is_csv),'uint8') typecast(int32(is_cm),'uint8'),...
                typecast(int32(is_rad),'uint8') typecast(int32(interpY),'uint8') tra_fileName]);

            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function saveTrackingData(obj,volNum,tra_fileName,save_as_matrix,save_as_rows,save_csv,convert_mm_to_cm,convert_deg_to_rad,interpY)
            % Saves a tracking data file for the given volume
            % only obj, volume, tra_fileName are required
            % save_as_matrix, save_as_rows, save_with_commas, convert_to_cm, convert_to_rad, interpolate are optional

            % volNum: The volume( numeric, index 0, set by cfg order) to save the tracking data for.
            % tra_fileName: The path and file name (.tra) to where tracking data will be saved
            % save_as_matrix: 1 to save as a matrix, 0 to save as xyzypr format
            % save_as_rows: 1 to save as rows, 0 to save as columns
            % save_csv: 1 to save with commas, 0 to save with spaces
            % convert_mm_to_cm: 1 to convert to cm, 0 to leave in mm
            % convert_deg_to_rad: 1 to convert to radians, 0 to leave in degrees
            % interpY: 1 to interpolate(Spline), 0 to leave as is

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
                save_csv = 1;
            end
            if nargin < 7
                convert_mm_to_cm = 0;
            end
            if nargin < 8
                convert_deg_to_rad = 0;
            end
            if nargin < 9
                interpY = 0;
            end
            fwrite(obj.socket_descriptor, [3 typecast(int32(volNum),'uint8') typecast(int32(save_as_matrix),'uint8') typecast(int32(save_as_rows),'uint8'),...
                typecast(int32(save_csv),'uint8') typecast(int32(convert_mm_to_cm),'uint8') typecast(int32(convert_deg_to_rad),'uint8'),...
                typecast(int32(interpY),'uint8') tra_fileName]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function loadFilters(obj, camera, filter_file)
            % Loads a filter file for the given camera

            % camera: the camera number to load . (index base 0)  -1 for all
            % filter_file: the filter file to load (.vie)

            if nargin < 3
                error('Not enough input arguments')
            end
            fwrite(obj.socket_descriptor, [4 typecast(int32(camera),'uint8') filter_file]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function setFrame(obj,frameNum)
            % Sets the frame number

            % frameNum: the frame number to set (index 0)

            if nargin < 2
                error('Not enough input arguments')
            end
            fwrite(obj.socket_descriptor, [5 typecast(int32(frameNum),'uint8')]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function pose = getPose(obj,volNum,frame)
            % Gets the pose for the given volume and frame

            % volNum: the volume number to get the pose for
            % frame: the frame number to get the pose for

            if nargin < 3
                error('Not enough input arguments')
            end
            fwrite(obj.socket_descriptor, [6 typecast(int32(volNum),'uint8') typecast(int32(frame),'uint8')]);
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

        function setPose(obj,volNum,frame,pose)
            % Sets the pose for the given volume and frame

            % volNum: the volume number to set the pose for
            % frame: the frame number to set the pose for
            % pose: the pose to set (set getPose)

            if nargin < 4
                error('Not enough input arguments')
            end
            fwrite(obj.socket_descriptor, [7 typecast(int32(volNum),'uint8') typecast(int32(frame),'uint8') typecast(double(pose),'uint8')]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function ncc = getNCC(obj, volNum, pose)
            % Gets the NCC for the given volume and pose

            % volNum: the volume number to get the NCC for
            % frameNum: the frame to get the NCC for

            if nargin < 3
                error('Not enough input arguments')
            end
            fwrite(obj.socket_descriptor, [8 typecast(int32(volNum),'uint8') typecast(double(pose),'uint8')]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
            ncc =nan(1,data(2));
            for i = 1:data(2)
                val = typecast(uint8(data(3 + (i-1)*8: 10+(i-1)*8)),'double');
                if(val == -99999)
                    val = NaN;
                end
                ncc(i) = val;
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

        function getImageCropped(obj, volNum, camera, frameNum)

            % not yet implemented

            % volNum: the volume number to get the image for
            % camera: the camera number to get the image for
            % frameNum: the frame number to get the image for

            if nargin < 4
                error('Not enough input arguments')
            end

            fwrite(obj.socket_descriptor, [10 typecast(int32(volNum),'uint8') typecast(int32(camera),'uint8') typecast(double(frameNum),'uint8')]);
            while obj.socket_descriptor.BytesAvailable == 0
                pause(1)
            end
            data = fread(obj.socket_descriptor, obj.socket_descriptor.BytesAvailable);
        end

        function optimizeFrame(obj, volNum, frameNum, repeats,max_itr,min_lim,max_lim,max_stall_itr,dframe,opt_method,cf_model)
            % Optimizes the given frame
            % only obj, volNum, and frame are required
            % all other parameters are optional

            % volNum: the volume number to optimize
            % frameNum: the frame number to optimize
            % repeats: the number of times to repeat the optimization
            % max_itr: the maximum number of iterations
            % min_lim: the minimum limit
            % max_lim: the maximum limit
            % max_stall_itr: the maximum number of iterations to stall
            % dframe: The amount of frames to skip
            % opt_method: The optimization method to use, 0 for Particle Swarm, 1 for Downhill Simplex
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
            fwrite(obj.socket_descriptor, [11 typecast(int32(volNum),'uint8') typecast(int32(frameNum),'uint8') typecast(int32(repeats),'uint8') typecast(int32(max_itr),'uint8') typecast(double(min_lim),'uint8') typecast(double(max_lim),'uint8') typecast(int32(max_stall_itr),'uint8') typecast(int32(dframe),'uint8') typecast(int32(opt_method),'uint8') typecast(int32(cf_model),'uint8')]);
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

        function trackingDialog(obj,volNum, startframe,endframe,repeats,max_itr,min_lim,max_lim,max_stall_itr,dframe,opt_method,cf_model)
            % Performs optimization on a range of frames
            % Only obj, volNum, startframe, and endframe are required
            % all other parameters are optional

            %volNum: volume to be optimized over the designated range
            % startframe: the first frame to optimize
            % endframe: the last frame to optimize
            % repeats: the number of times to repeat the optimization
            % max_itr: the maximum number of iterations
            % min_lim: the minimum limit
            % max_lim: the maximum limit
            % max_stall_itr: the maximum number of iterations to stall
            % dframe: The amount of frames to skip
            % opt_method: The optimization method to use, 0 for Particle Swarm, 1 for Downhill Simplex
            % cf_model: The cost function model to use, 0 for NCC (Bone Models), 1 for Sum of Absolute Differences (Implant Models)

            if nargin < 4
                error('Not enough input arguments')
            end
            if nargin < 5
                repeats = 1;
            end
            if nargin < 6
                max_itr = 1000;
            end
            if nargin < 7
                min_lim = -3.0;
            end
            if nargin < 8
                max_lim = 3.0;
            end
            if nargin < 9
                max_stall_itr = 25;
            end
            if nargin < 10
                dframe = 1;
            end
            if nargin < 11
                opt_method = 0;
            end
            if nargin < 12
                cf_model = 0;
            end

            for i = startframe:endframe
                obj.setFrame(i);
                if i ~= 0
                    pose = obj.getPose(volNum,i-1);
                    obj.setPose(volNum,i,pose);
                end
                obj.optimizeFrame(volNum,i,repeats,max_itr,min_lim,max_lim,max_stall_itr,dframe,opt_method,cf_model);
            end
        end

        function ncc_out = getNCC_Sum(obj,volNum,pose)
            % Gets the sum of the NCC for the given volNum and pose

            % volNum: the volume number to get the NCC for
            % pose: the pose to get the NCC for

            if nargin < 3
                error('Not enough input arguments')
            end
            ncc = obj.getNCC(volNum,pose);
%             ncc = zeros(1,data(2));
%             for i = 1:data(2)
%                 val = typecast(uint8(data(3 + (i-1)*8: 10+(i-1)*8)),'double') ;
%                 if(val < 0)
%                     val = 1e3;
%                 end
%                 ncc(i) = val;
%             end

            ncc_out  = ncc(1) + ncc(2);
        end

        function ncc_out = getNCC_This_Frame(obj,volNum,frameNum)
            % Gets the NCC for the given volNum and frame

            % volNum: the volume number to get the NCC for
            % frame: the frame to get the NCC for

            if nargin < 3
                error('Not enough input arguments')
            end
            pose = obj.getPose(volNum,frameNum);
            ncc = obj.getNCC(volNum,pose);
%             ncc = nan(1,length(data));
%             for i = 1:length(data)
%                 val = typecast(uint8(data(3 + (i-1)*8: 10+(i-1)*8)),'double') ;
%                 if(val == -99999)
%                     val = NaN;
%                 end
%                 ncc(i) = val;
%             end

            ncc_out  = [ncc(1),ncc(2),ncc(1)*ncc(2)];
        end
    end
end