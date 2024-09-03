function idx_save = check_save_idx(folder_path, postname)
    % Get a list of files in the folder
    files = dir([folder_path, '/*', postname, '.mat']);
    
    % Initialize an array to hold the indices
    indices = [];

    % Loop through each file and extract the index
    for i = 1:length(files)
        % Get the current filename without extension
        [~, name, ~] = fileparts(files(i).name);
        
        % Check if the filename ends with the postname and has a numeric prefix
        if startsWith(name, '_') % Skip if no numeric prefix
            continue;
        end
        prefix = extractBefore(name, postname);
        
        % Check if the prefix is numeric
        idx = str2double(prefix);
        if ~isnan(idx)
            indices(end+1) = idx; %#ok<AGROW> % Add index to the array
        end
    end
    
    % Determine the next index
    if isempty(indices)
        idx_save = 1; % Start with 1 if no files are found
    else
        idx_save = max(indices) + 1; % Increment the highest index
    end
end