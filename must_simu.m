function must_simu(diffusor_fn, output_fn)
    
    param = getparam('P4-2v');
    
    param.TXapodization = cos(linspace(-1,1,64)*pi/2);
    
    tilt = deg2rad(linspace(-20,20,7)); % tilt angles in rad
    txdel = cell(7,1); % this cell will contain the transmit delays
    
    for k = 1:7
        txdel{k} = txdelay(param,tilt(k),deg2rad(60));
    end
    
    I = nrrdread(diffusor_fn);
    % Pseudorandom distribution of scatterers (depth is 15 cm)
    [x,y,z,RC] = genscat([NaN 15e-2],1540/param.fc,I);
    
    RF = cell(7,1); % this cell will contain the RF series
    param.fs = 4*param.fc; % sampling frequency in Hz
    
    for k = 1:7       
        RF{k} = simus(x,y,z,RC,txdel{k},param);
    end    
    
    IQ = cell(7,1);  % this cell will contain the I/Q series
    
    for k = 1:7
        IQ{k} = rf2iq(RF{k},param.fs,param.fc);
    end    
    
    [xi,zi] = impolgrid([256 256],15e-2,deg2rad(80),param);
    
    bIQ = zeros(256,256,7);  % this array will contain the 7 I/Q images
    
    for k = 1:7
        bIQ(:,:,k) = das(IQ{k},xi,zi,txdel{k},param);
    end    
    
    bIQ = tgc(bIQ);
    
    
    cIQ = sum(bIQ,3); % this is the compound beamformed I/Q
    I = bmode(cIQ,50); % log-compressed image

    fig = figure('Visible', 'off');

    % Set figure and paper position
    [outputHeight, outputWidth] = size(I);
    outputHeight = outputHeight + 64
    outputWidth = outputWidth + 64
    set(fig, 'Units', 'pixels', 'Position', [0 0 outputWidth outputHeight]);
    set(fig, 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
    set(fig, 'Color', [0 0 0]);  % Set background color to black
    
    pcolor(xi*1e2,zi*1e2,I)
    shading interp, colormap gray

    % Adjust axes to fill figure
    ax = gca;
    set(ax, 'InnerPosition', [0 0 1 1]);

    axis equal ij;
    axis off;

    fig.Visible = 'on';

    frame = getframe(ax);
    imageData = frame2im(frame);

    width = 255
    height = 255
    imageData = imcrop(imageData, [32 32 width height]);

    save(output_fn, "I", "IQ", "RF", "imageData");

    return;
    
end