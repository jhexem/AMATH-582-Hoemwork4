function dcData = dc_wavelet(dcfile) 
% Input: dog/cat matrix
% Output: edge detection

    [m,n] = size(dcfile); % 4096 x 80
    pxl = sqrt(m); % cus images are square
    nw = m/4; % wavelet resolution cus downsampling
    dcData = zeros(nw,n);
    
    for k = 1:n
        X = im2double(reshape(dcfile(:,k),pxl,pxl));
        [~,cH,cV,~]=dwt2(X,'haar'); % only want horizontal and vertical
        cod_cH1 = rescale(abs(cH)); % horizontal rescaled
        cod_cV1 = rescale(abs(cV)); % vertical rescaled
        cod_edge = cod_cH1+cod_cV1; % edge detection
        dcData(:,k) = reshape(cod_edge,nw,1);
    end
end
