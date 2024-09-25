function idx_save = check_save_idx_folder(folder_path, prefixname, postname)
    % Get a list of files in the folder
    files = dir(folder_path);
    
    % Initialize an array to hold the indices
    indices = [];

    % Loop through each file and extract the index
    for i = 1:length(files)
        % Get the current filename without extension
        [~, name, ~] = fileparts(files(i).name);
        
        % Check if the filename starts with the prefixname and ends with the postname
        if startsWith(name, prefixname) && endsWith(name, postname)
            % Extract the part between the prefixname and postname
            number_str = extractBetween(name, prefixname, postname);
            
            % Convert the extracted string to a number
            idx = str2double(number_str{1});
            
            % If it is a valid number, store it in the indices array
            if ~isnan(idx)
                indices(end+1) = idx; %#ok<AGROW>
            end
        end
    end
    
    % Determine the next index
    if isempty(indices)
        idx_save = 1; % Start with 1 if no files are found
    else
        idx_save = max(indices) + 1; % Increment the highest index
    end
end