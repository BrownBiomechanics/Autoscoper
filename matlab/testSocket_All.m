volume = 0;

%open Connection
autoscoper_socket = openConnection('127.0.0.1');

%load Trial
loadTrial(autoscoper_socket, 'D:\data\Autoscoper Development\test - Mayacam1 - Copy.cfg');

%Load tracking
loadTacking(autoscoper_socket, volume, 'D:\data\Autoscoper Development\tracking.tra');

%set Background
setBackground(autoscoper_socket,0.3);

%load filter settings (optional)
%loadFilters(autoscoper_socket,0,'D:\data\Autoscoper Development\Trial001\tracking_settings.cfg');
%loadFilters(autoscoper_socket,1,'D:\data\Autoscoper Development\Trial001\tracking_settings.cfg');

for frame=1:10
    %set frame
    setFrame(autoscoper_socket,frame);

    %get pose for the frame
    pose = getPose(autoscoper_socket,volume,frame)

    %%%%%%%%%%% Start Optimization
        %run optimizer here and repeatidly get the ncc
    ncc = getNCC(autoscoper_socket,volume,pose)
    %%%%%%%%%%%End Optimization

    %At the end set the best pose
    setPose(autoscoper_socket,volume,frame,pose);
end

%If everything is done: Save tracking
saveTracking(autoscoper_socket,volume,'D:\data\Autoscoper Development\tracking_save.tra');