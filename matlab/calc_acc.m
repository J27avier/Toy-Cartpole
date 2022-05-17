function accelerations =  calc_acc(x, x_dot, theta, theta_dot, force)
    gravity = 9.8;
    masscart = 1.0;
    masspole = 0.1;
    total_mass = masspole + masscart;
    length = 0.5;  % actually half the pole's length
    polemass_length = masspole * length;
    force_mag = 10.0;
    tau = 0.02;  % seconds between state updates

    costheta = cos(theta);
    sintheta = sin(theta);

    temp = ( force + polemass_length * theta_dot^2 * sintheta) / total_mass;

    thetaacc = (gravity * sintheta - costheta * temp) / (...
        length * (4.0 / 3.0 - masspole * costheta^2 / total_mass));

    xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    % Integration
    x = x + tau * x_dot;
    x_dot = x_dot + tau * xacc;
    theta = theta + tau * theta_dot;
    theta_dot = theta_dot + tau * thetaacc;

    accelerations =  [xacc; thetaacc];
endfunction
