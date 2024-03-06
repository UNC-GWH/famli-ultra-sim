function must_simu_temp(diffusor_fn, output_fn)
    
    param = getparam('P4-2v');    
    param.TXapodization = cos(linspace(-1,1,64)*pi/2);
    param.fs = 4*param.fc; % sampling frequency in Hz
    
    [xi,zi] = impolgrid([256 256],15e-2,deg2rad(80),param);

    load(output_fn);

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