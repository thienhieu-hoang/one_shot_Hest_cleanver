% Define the file name and open it for writing ('w' mode creates the file if it doesn't exist)
fileID = fopen('DeepMIMO_Data/Static_BS16/freq_symb_1ant_612sub_ver4/readme.txt', 'w');  

% Check if the file was opened successfully
if fileID == -1
    error('Cannot open or create the file for writing.');
end

% Define the content to write to the file
content = ['This is the readme file.', newline, ...
           'Here you can add descriptions about the project.', newline, ...
           'Instructions for running the code:', newline, ...
           '1. Step one', newline, ...
           '2. Step two', newline, ...
           '3. Step three', newline];

% Write the content to the file
fprintf(fileID, '%s', content);

% Close the file
fclose(fileID);

disp('readme.txt file created successfully.');