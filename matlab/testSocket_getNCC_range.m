volume = 0;
autoscoper_socket = openConnection('127.0.0.1');
first_frame = 0;
last_frame = 100;

%
ncc_values = [];
for frame = first_frame:last_frame
    %set frame
    setFrame(autoscoper_socket,frame);

    %get pose for the frame
    curPose = getPose(autoscoper_socket,volume,frame);

    %get ncc values
    ncc_values = getNCC(autoscoper_socket,volume,curPose);
end
ncc_values(:,3) = ncc_values(:,1).*ncc_values(:,2);