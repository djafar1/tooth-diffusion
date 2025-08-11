function inpaintedVolume = fillToothCavity(imageVolume, labelVolume, toothID)

toothMask = imdilate(labelVolume == toothID, strel('sphere', 2)); % expanded tooth mask
teethMask = imdilate(labelVolume > 0, strel('sphere', 2)); % expanded teeth mask
V1 = imageVolume; % initialized image volume with missing tooth
V1(teethMask) = NaN; % remove all teeth from image volume
V1 = imgaussfilt3(V1, 0.5); % smooth image volume without tooth

missing = isnan(V1); % indices of all missing teeth
[~, idxNear] = bwdist(~missing, 'cityblock'); % distance transform of tooth-less binary volume
V2 = imageVolume; % initialized inpainted image volume
V2(teethMask) = V2(idxNear(teethMask)); % filled missing tooth values using the nearest values
V2 = imgaussfilt3(V2, 1); % smooth inpainted image volume

inpaintedVolume = imageVolume; % initialized inpainted image volume
inpaintedVolume(toothMask) = V2(toothMask); % blended image volume in the hole region only
