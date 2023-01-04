function [ncc] = getNCC(autoscoper_socket,volume,pose)
    %GETNCC Summary of this function goes here
    %   Detailed explanation goes here

    fwrite(t,[autoscoper_socket typecast(int32(volume),'uint8') typecast(double(pose),'uint8')]);
    while autoscoper_socket.BytesAvailable == 0
        pause(1)
    end
    data = fread(autoscoper_socket, autoscoper_socket.BytesAvailable);

    %-99999 is nsot tracked
    %returned values is 1 - ncc, so 0 is best, 2 is worst
    ncc = {};
    for i = 1:data(2)
        val = typecast(uint8(data(3 + (i-1)*8: 10+(i-1)*8)),'double')
        if(val == -99999)
            val = NaN;
        end
        ncc = [ncc val];
    end
end

