function mkfile
%make file

%     filename = fullfile(tempdir, 'talk_answer.dat');
    filename = 'talk_answer.dat';
%     myData = uint8(1:10)';
    [f, msg] = fopen(filename, 'wb');
    if f~= -1
         fwrite(f, zeros(1,256), 'uint8');
%         fwrite(f,myData,'uint8');
        fclose(f);
    else
        error('MATLAB:demo:send:cannotOpenFile', ...
            'Cannot open file "%s": %s.', filename, msg);
    end
end