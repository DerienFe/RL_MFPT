%% Choose unbiasing method

clear all
clc

Num_eigvec = 5;

for ii = 1:1:Num_eigvec

    load(['data_eigenvector_' num2str(ii), '_.mat'])

    bias_order_win = zeros(num_windows,n_states);
    for iii = 1:1:num_windows

        bias_order_win(iii,:) = xx;

    end

    for j = 1:1:Num_exper

        Record_states_sample = Record_states(:,:,j);
        prob_dist{j} = DHAM_unbias(Record_states_sample, x_eq, force_constant, kT, n_states, bias_order_win, cutoff);

    end

    %% Mean probability distribution with error bars

    prob_record = zeros(Num_exper, n_states);
    for jj = 1:1:Num_exper

        prob_record(jj,:) = prob_dist{jj};

    end

    figure
    box on
    hold on
    color = [0.00,0.45,0.74];
    plot(1:1:n_states,pi,'k')
    plot(1:1:n_states, mean(prob_record),'MarkerEdgeColor',color, 'MarkerFaceColor','none')
    errorbar(1:1:n_states, mean(prob_record), std(prob_record)./sqrt(size(prob_record,2)),'LineStyle','none', 'Color',color);
    xlabel('nodes', 'interpreter','latex')
    ylabel('probablity distribution')
    legend('Exact','Empirical')
    savefig(['data_eigenvector_' num2str(ii), '_prob_dist_.fig'])
    close all
    %% Mean Free energy profile with error bars

    free_energy = real(-log(prob_record));
    free_energy(isinf(free_energy)) = 100;

    for i = 1:1:size(free_energy,2)

        mean_free_energy(i) = mean(free_energy(:,i));
        ste_free_energy(i)  = std(free_energy(:,i));

    end

    RMSE_free_energy = sqrt(mean ( ( mean_free_energy - F./kT ).^2 ));

    EE = 2*(ste_free_energy./(mean_free_energy - F./kT)).*( (mean_free_energy - F./kT).^2 ); % error propagation for raising to the power. Look at line 3 in table 3.1 of Student's guide to data and error analysis. Note that you are settng x = 1 in line 3.
    EE = sqrt( sum(EE.^2) )./numel(mean_free_energy); % error propagation for addition: look at line 1 of table 3.1.
    RMSE_error = abs(0.5.*(EE./(RMSE_free_energy.^2)).*RMSE_free_energy); % eror propagation for raising to the power of 0.5. look at line 3 of table 3.1

    figure
    box on
    hold on
    color = [0.00,0.45,0.74];
    plot(1:1:n_states,F./kT,'k')
    plot(1:1:n_states, mean_free_energy,'MarkerEdgeColor',color, 'MarkerFaceColor','none')
    errorbar(1:1:n_states, mean_free_energy, ste_free_energy,'LineStyle','none', 'Color',color);
    xlabel('nodes', 'interpreter','latex')
    ylabel('Free energy')
    legend('Exact','Empirical')
    title(['RMSE: ' num2str(mean(RMSE_free_energy))], 'interpreter','latex')

    savefig(['data_eigenvector_' num2str(ii), '_free_energy_.fig'])
    save(['data_eigenvector_' num2str(ii), '_.mat'])
    clearvars -except Num_eigvec ii
    close all

end

