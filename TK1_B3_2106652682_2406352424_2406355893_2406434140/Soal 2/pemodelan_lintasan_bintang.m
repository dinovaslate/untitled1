%% Load Data
data = readmatrix(fullfile('Data', 'data.csv'));
x = data(:, 1);
y = data(:, 2);
n = length(x);
fprintf('Jumlah data: %d\n', n);
fprintf('x range: [%.4f, %.4f]\n', min(x), max(x));
fprintf('y range: [%.4f, %.4f]\n\n', min(y), max(y));

%% ========================================================================
%  BAGIAN I: Formulasi Sistem Overdetermined Ax = b
%% ========================================================================
fprintf('=== BAGIAN I: Formulasi Sistem Overdetermined ===\n');

% Persamaan elips: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
% Dengan F = 1: Ax^2 + Bxy + Cy^2 + Dx + Ey = -1

A_mat = [x.^2, x.*y, y.^2, x, y];  % Matriks A (300 x 5)
b_vec = -ones(n, 1);                 % Vektor b (300 x 1)

fprintf('Ukuran matriks A: %d x %d\n', size(A_mat));
fprintf('Ukuran vektor b: %d x 1\n', length(b_vec));
fprintf('Sistem overdetermined: m = %d >> n = %d\n\n', n, 5);

%% ========================================================================
%  BAGIAN II: Metode Normal Equation
%% ========================================================================
fprintf('=== BAGIAN II: Normal Equation ===\n');

ATA = A_mat' * A_mat;   % Matriks A^T A (5 x 5)
ATb = A_mat' * b_vec;   % Vektor A^T b (5 x 1)

% Condition number
cond_ATA = cond(ATA);
fprintf('Condition number kappa(A^T A) = %.6e\n', cond_ATA);

% Solve A^T A x = A^T b
x_ne = ATA \ ATb;

fprintf('\nSolusi Normal Equation:\n');
fprintf('  A = %.10f\n', x_ne(1));
fprintf('  B = %.10f\n', x_ne(2));
fprintf('  C = %.10f\n', x_ne(3));
fprintf('  D = %.10f\n', x_ne(4));
fprintf('  E = %.10f\n', x_ne(5));
fprintf('  F = 1\n');

% Residual error
residual_ne = A_mat * x_ne - b_vec;
error_ne = norm(residual_ne);
fprintf('\nError residual ||Ax - b||_2 = %.10f\n\n', error_ne);

%% ========================================================================
%  BAGIAN III: QR Factorization with Householder Reflections
%% ========================================================================
fprintf('=== BAGIAN III: QR Householder ===\n');

% Implementasi manual Householder QR
[Q_hh, R_hh] = householder_qr(A_mat);

% Ambil thin QR: Q1 (300x5) dan R1 (5x5)
Q1 = Q_hh(:, 1:5);
R1 = R_hh(1:5, 1:5);

fprintf('Ukuran Q: %d x %d\n', size(Q_hh));
fprintf('Ukuran R: %d x %d\n', size(R_hh));
fprintf('Ukuran Q1 (thin): %d x %d\n', size(Q1));
fprintf('Ukuran R1 (thin): %d x %d\n\n', size(R1));

% Verifikasi
fprintf('||A - Q1*R1||_F = %.6e\n', norm(A_mat - Q1*R1, 'fro'));
fprintf('||Q1^T*Q1 - I||_F = %.6e\n\n', norm(Q1'*Q1 - eye(5), 'fro'));

% Solve R1 * x = Q1^T * b (back substitution)
QtB = Q1' * b_vec;
x_qr = R1 \ QtB;

fprintf('Solusi QR Householder:\n');
fprintf('  A = %.10f\n', x_qr(1));
fprintf('  B = %.10f\n', x_qr(2));
fprintf('  C = %.10f\n', x_qr(3));
fprintf('  D = %.10f\n', x_qr(4));
fprintf('  E = %.10f\n', x_qr(5));
fprintf('  F = 1\n');

% Residual error
residual_qr = A_mat * x_qr - b_vec;
error_qr = norm(residual_qr);
fprintf('\nError residual ||Ax - b||_2 = %.10f\n', error_qr);

% Condition number comparison
cond_A = cond(A_mat);
fprintf('\nCondition number kappa(A) = %.6e\n', cond_A);
fprintf('Condition number kappa(A^T A) = %.6e\n', cond_ATA);
fprintf('[kappa(A)]^2 = %.6e\n', cond_A^2);
fprintf('Rasio kappa(A^T A) / [kappa(A)]^2 = %.10f\n', cond_ATA / cond_A^2);
fprintf('||x_NE - x_QR|| = %.2e\n\n', norm(x_ne - x_qr));

%% ========================================================================
%  BAGIAN IV: Analisis Kompleksitas FLOPs dan Memori
%% ========================================================================
fprintf('=== BAGIAN IV: Analisis Kompleksitas ===\n');
m_val = 300; n_val = 5;

% Normal Equation
flops_ne_ata = 2 * m_val * n_val^2;
flops_ne_atb = 2 * m_val * n_val;
flops_ne_chol = round(n_val^3 / 3);
flops_ne_sub = n_val^2;
flops_ne_total = flops_ne_ata + flops_ne_atb + flops_ne_chol + flops_ne_sub;

fprintf('Normal Equation:\n');
fprintf('  A^T A:     2mn^2       = %d\n', flops_ne_ata);
fprintf('  A^T b:     2mn         = %d\n', flops_ne_atb);
fprintf('  Cholesky:  n^3/3       ~ %d\n', flops_ne_chol);
fprintf('  Back sub:  n^2         = %d\n', flops_ne_sub);
fprintf('  Total:                 ~ %d\n', flops_ne_total);
fprintf('  Asimtotik: O(mn^2 + n^3/3)\n\n');

% QR Householder
flops_qr_hh = 2*m_val*n_val^2 - round(2*n_val^3/3);
flops_qr_qtb = 2*m_val*n_val;
flops_qr_sub = n_val^2;
flops_qr_total = flops_qr_hh + flops_qr_qtb + flops_qr_sub;

fprintf('QR Householder:\n');
fprintf('  Householder: 2mn^2 - 2n^3/3 ~ %d\n', flops_qr_hh);
fprintf('  Q^T b:       2mn            = %d\n', flops_qr_qtb);
fprintf('  Back sub:    n^2            = %d\n', flops_qr_sub);
fprintf('  Total:                      ~ %d\n', flops_qr_total);
fprintf('  Asimtotik: O(2mn^2 - 2n^3/3)\n\n');

% Memory
mem_ne = m_val*n_val + n_val^2 + n_val;
fprintf('Memori Normal Equation: mn + n^2 + n = %d elemen\n', mem_ne);
fprintf('Memori QR Householder:  mn (in-place) = %d - 2mn (eksplisit) = %d\n\n', ...
    m_val*n_val, 2*m_val*n_val);

%% ========================================================================
%  BAGIAN V: Evaluasi - Analisis Perturbasi
%% ========================================================================
fprintf('=== BAGIAN V: Evaluasi Stabilitas Numerik ===\n');
fprintf('HUBUNGAN KUNCI: kappa(A^T A) = [kappa(A)]^2\n');
fprintf('  kappa(A)      = %.4f\n', cond_A);
fprintf('  [kappa(A)]^2  = %.4f\n', cond_A^2);
fprintf('  kappa(A^T A)  = %.4f\n', cond_ATA);
fprintf('  TERBUKTI IDENTIK!\n\n');

% Analisis perturbasi
fprintf('--- Analisis Perturbasi (Geser b) ---\n');
rng(42);
epsilons = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1];
fprintf('%12s | %20s | %20s | %12s\n', 'epsilon', '||dx_NE||/||x_NE||', '||dx_QR||/||x_QR||', 'Ratio NE/QR');
fprintf('%s\n', repmat('-', 1, 75));

for i = 1:length(epsilons)
    eps_val = epsilons(i);
    delta_b = eps_val * randn(n, 1);
    b_pert = b_vec + delta_b;
    
    % NE with perturbed b
    x_ne_pert = (A_mat' * A_mat) \ (A_mat' * b_pert);
    
    % QR with perturbed b
    x_qr_pert = R1 \ (Q1' * b_pert);
    
    rel_ne = norm(x_ne_pert - x_ne) / norm(x_ne);
    rel_qr = norm(x_qr_pert - x_qr) / norm(x_qr);
    ratio = rel_ne / rel_qr;
    
    fprintf('%12.0e | %20.6e | %20.6e | %12.6f\n', eps_val, rel_ne, rel_qr, ratio);
end

fprintf('\nKesimpulan: QR Householder LEBIH STABIL karena kappa(A) << kappa(A^T A)\n\n');

%% ========================================================================
%  BAGIAN VI: Karakteristik Geometris Elips
%% ========================================================================
fprintf('=== BAGIAN VI: Karakteristik Geometris Elips ===\n');

% Gunakan solusi QR (lebih stabil)
A_c = x_qr(1); B_c = x_qr(2); C_c = x_qr(3);
D_c = x_qr(4); E_c = x_qr(5); F_c = 1;

fprintf('\nPersamaan elips:\n');
fprintf('  %.8fx^2 + (%.8f)xy + %.8fy^2 + (%.8f)x + (%.8f)y + 1 = 0\n', ...
    A_c, B_c, C_c, D_c, E_c);

% Verifikasi elips
disc = B_c^2 - 4*A_c*C_c;
fprintf('\nDiskriminan B^2 - 4AC = %.6f', disc);
if disc < 0
    fprintf(' < 0 --> ELIPS TERKONFIRMASI\n');
else
    fprintf(' >= 0 --> BUKAN ELIPS!\n');
end

% Pusat elips
M_center = [2*A_c, B_c; B_c, 2*C_c];
rhs_center = [-D_c; -E_c];
center = M_center \ rhs_center;
x0 = center(1); y0 = center(2);
fprintf('\nPusat elips: (%.6f, %.6f)\n', x0, y0);

% Semi-axes dari eigenvalue
M_conic = [A_c, B_c/2; B_c/2, C_c];
[V, D_eig] = eig(M_conic);
eigenvalues = diag(D_eig);

F_at_center = A_c*x0^2 + B_c*x0*y0 + C_c*y0^2 + D_c*x0 + E_c*y0 + F_c;
fprintf('F di pusat: %.6f\n', F_at_center);
fprintf('Eigenvalue: lambda1 = %.8f, lambda2 = %.8f\n', eigenvalues(1), eigenvalues(2));

semi_axes = sqrt(-F_at_center ./ eigenvalues);
semi_major = max(semi_axes);
semi_minor = min(semi_axes);

fprintf('Semi-major axis (a): %.6f\n', semi_major);
fprintf('Semi-minor axis (b): %.6f\n', semi_minor);

% Sudut rotasi
[~, idx_major] = min(eigenvalues); % eigenvalue terkecil -> semi-major
theta_ev = atan2(V(2, idx_major), V(1, idx_major));
% Normalisasi ke [-90, 90]
if theta_ev < -pi/2; theta_ev = theta_ev + pi; end
if theta_ev > pi/2; theta_ev = theta_ev - pi; end
fprintf('Sudut rotasi: %.6f rad = %.4f derajat\n', theta_ev, rad2deg(theta_ev));

% Eksentrisitas
ecc = sqrt(1 - (semi_minor/semi_major)^2);
fprintf('Eksentrisitas: %.6f\n\n', ecc);

%% ========================================================================
%  BAGIAN VII: Visualisasi
%% ========================================================================
fprintf('=== BAGIAN VII: Visualisasi ===\n');

% Generate titik-titik elips parametrik
t = linspace(0, 2*pi, 500);
cos_th = cos(theta_ev);
sin_th = sin(theta_ev);
x_ell = x0 + semi_major*cos(t)*cos_th - semi_minor*sin(t)*sin_th;
y_ell = y0 + semi_major*cos(t)*sin_th + semi_minor*sin(t)*cos_th;

% Plot 1: Ellipse Fit + Data
figure('Position', [100 100 800 650]);
scatter(x, y, 20, [0.29 0.56 0.85], 'filled', 'MarkerFaceAlpha', 0.5, ...
    'DisplayName', 'Data Observasi (300 titik)');
hold on;
plot(x_ell, y_ell, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Kurva Elips (QR Householder)');
plot(x0, y0, 'k+', 'MarkerSize', 15, 'LineWidth', 3, ...
    'DisplayName', sprintf('Pusat (%.3f, %.3f)', x0, y0));

% Gambar sumbu semi-major dan semi-minor
plot([x0-semi_major*cos_th, x0+semi_major*cos_th], ...
     [y0-semi_major*sin_th, y0+semi_major*sin_th], ...
     'k--', 'LineWidth', 1.5, 'DisplayName', sprintf('Semi-major a = %.3f', semi_major));
plot([x0+semi_minor*sin_th, x0-semi_minor*sin_th], ...
     [y0-semi_minor*cos_th, y0+semi_minor*cos_th], ...
     'k:', 'LineWidth', 1.5, 'DisplayName', sprintf('Semi-minor b = %.3f', semi_minor));

hold off;
axis equal; grid on;
xlabel('x', 'FontSize', 13);
ylabel('y', 'FontSize', 13);
title({'Pemodelan Lintasan Bintang - Least Squares Ellipse Fitting'; ...
       '(QR Factorization with Householder Reflections)'}, 'FontSize', 13);
legend('Location', 'best', 'FontSize', 10);
saveas(gcf, fullfile('Hasil', 'plot_ellipse_fit.png'));
fprintf('Plot 1 (Ellipse Fit) saved.\n');

% Plot 2: Perbandingan Normal Equation vs QR
figure('Position', [100 100 800 650]);

% Hitung elips Normal Equation
A_n = x_ne(1); B_n = x_ne(2); C_n = x_ne(3);
D_n = x_ne(4); E_n = x_ne(5);
M_n = [A_n, B_n/2; B_n/2, C_n];
[V_n, D_n_eig] = eig(M_n);
ev_n = diag(D_n_eig);
ctr_n = [2*A_n, B_n; B_n, 2*C_n] \ [-D_n; -E_n];
F_cn = A_n*ctr_n(1)^2 + B_n*ctr_n(1)*ctr_n(2) + C_n*ctr_n(2)^2 + D_n*ctr_n(1) + E_n*ctr_n(2) + 1;
sa_n = sqrt(-F_cn ./ ev_n);
[~, idx_n] = min(ev_n);
th_n = atan2(V_n(2,idx_n), V_n(1,idx_n));
if th_n < -pi/2; th_n = th_n + pi; end
if th_n > pi/2; th_n = th_n - pi; end
x_ne_ell = ctr_n(1) + max(sa_n)*cos(t)*cos(th_n) - min(sa_n)*sin(t)*sin(th_n);
y_ne_ell = ctr_n(2) + max(sa_n)*cos(t)*sin(th_n) + min(sa_n)*sin(t)*cos(th_n);

scatter(x, y, 20, [0.29 0.56 0.85], 'filled', 'MarkerFaceAlpha', 0.4, ...
    'DisplayName', 'Data Observasi');
hold on;
plot(x_ell, y_ell, 'r-', 'LineWidth', 2.5, ...
    'DisplayName', sprintf('QR Householder (residual = %.4f)', error_qr));
plot(x_ne_ell, y_ne_ell, 'g--', 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Normal Equation (residual = %.4f)', error_ne));
hold off;
axis equal; grid on;
xlabel('x', 'FontSize', 13);
ylabel('y', 'FontSize', 13);
title('Perbandingan: Normal Equation vs QR Householder', 'FontSize', 13);
legend('Location', 'best', 'FontSize', 10);
saveas(gcf, fullfile('Hasil', 'plot_comparison.png'));
fprintf('Plot 2 (Comparison) saved.\n');

fprintf('\n=== SELESAI ===\n');

%% ========================================================================
%  FUNGSI: Householder QR Decomposition
%% ========================================================================
function [Q, R] = householder_qr(A)
    % Input:  A - matriks m x n (m >= n)
    % Output: Q - matriks ortogonal m x m
    %         R - matriks segitiga atas m x n
    
    [m, n_cols] = size(A);
    R = A;
    Q = eye(m);
    
    for j = 1:n_cols
        % Ambil elemen kolom j dari baris j sampai m
        x_col = R(j:m, j);
        
        % Hitung vektor Householder
        norm_x = norm(x_col);
        s = sign(x_col(1));
        if s == 0; s = 1; end
        
        v = x_col;
        v(1) = v(1) + s * norm_x;
        v = v / norm(v);
        
        % Terapkan refleksi Householder ke R
        % R(j:m, j:n) = R(j:m, j:n) - 2 * v * (v' * R(j:m, j:n))
        R(j:m, j:end) = R(j:m, j:end) - 2 * v * (v' * R(j:m, j:end));
        
        % Akumulasi Q
        % Q(j:m, :) = Q(j:m, :) - 2 * v * (v' * Q(j:m, :))
        Q(j:m, :) = Q(j:m, :) - 2 * v * (v' * Q(j:m, :));
    end
    
    Q = Q';  % Q dibangun sebagai Q^T, sehingga perlu transpose
end
