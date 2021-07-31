[mix, fs] = audioread('./audio/tooth_and_claw.wav');

mask_type = 0; % 1 = binary, 0 = soft
direct = 0;
display_cwt = 0;
display_specgram = 1;

beta = 2.0; % hard margin
p = 2; % soft/wiener filter power

[m, f, g, fshifts] = cqt(mix);
M = abs(m);

H = movmedian(M, 12, 2);
P = movmedian(M, 500, 1);

if mask_type == 1
    % hard mask, like drieger et al 2014
    Mh = (H./(P + eps)) > beta;
    Mp = (P./(H + eps)) >= beta;
else
    % soft mask fitzgerald 2010
    Hp = H.^p;
    Pp = P.^p;
    total = Hp + Pp;
    Mh = Hp./total;
    Mp = Pp./total;
end

if direct == 1
    h = H;
    p = P;
else
    h = Mh .* m;
    p = Mp .* m;
end

hrecon = icqt(h, g, fshifts);
precon = icqt(p, g, fshifts);

hrecon = hrecon/max(abs(hrecon));
precon = precon/max(abs(precon));

audiowrite('./p_recon.wav', precon, fs);
audiowrite('./h_recon.wav', hrecon, fs);

if display_cwt == 1
    figure; cqt(mix); title('Mix');
    figure; cqt(precon); title('Perc recon');
    figure; cqt(hrecon); title('Harm recon');
end

if display_specgram == 1
    figure; spectrogram(mix,1024,512,1024,fs,"yaxis"); title('Mix');
    figure; spectrogram(precon,1024,512,1024,fs,"yaxis"); title("Perc recon");
    figure; spectrogram(hrecon,1024,512,1024,fs,"yaxis"); title("Harm recon");
end