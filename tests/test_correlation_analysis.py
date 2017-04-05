# global imports
import unittest
import numpy as np

# local imports
import correlation_toolbox.helper as cthlp
import correlation_toolbox.correlation_analysis as ctana


class TestCorrelationAnalysis(unittest.TestCase):

    def setUp(self):
        np.random.seed(12345)
        self.rate = 30.  # (Hz)
        self.T = 3e4  # (ms)
        self.N = 100
        self.p = 0.6  # percentage of neurons active
        self.Neff = int(self.p * self.N)
        self.cc = 0.3
        self.v_mu = -50.
        self.v_var = 3.
        self.tbin = 2.
        self.Df = 1.
        self.fcut = 30.
        self.tau_max = 100.*10

    def test_mean(self):
        v = np.array([np.random.normal(i, self.v_var, self.T)
                     for i in range(self.N)])
        v_units = ctana.mean(v, units=True)
        v_time = ctana.mean(v, time=True)
        v_units_time = ctana.mean(v, units=True, time=True)
        v_units_true = np.mean(v, axis=0)
        v_time_true = np.arange(0., self.N)
        v_units_time_true = np.mean(v_time_true)
        self.assertTrue(abs(np.sum(v_units_true - v_units)) < 1e-16)
        self.assertTrue(abs(np.sum(v_time_true - v_time)) < self.N * 1e-2)
        self.assertTrue(abs(v_units_time_true - v_units_time) < 1e-2)

    def test_compound_mean(self):
        v = np.array([np.random.normal(i, self.v_var, self.T)
                     for i in range(self.N)])
        v_compound = ctana.compound_mean(v)
        v_compound_true = np.mean(np.sum(v, axis=0))
        self.assertTrue(abs(v_compound_true - v_compound) < 1e-15)

    def test_variance(self):
        v = np.array([np.random.normal(self.v_mu, np.sqrt(i + 1), self.T)
                     for i in range(self.N)])
        v_units = ctana.variance(v, units=True)
        v_time = ctana.variance(v, time=True)
        v_units_time = ctana.variance(v, units=True, time=True)
        v_units_true = np.var(v, axis=0)
        v_time_true = np.arange(0., self.N) + 1
        v_units_time_true = np.mean(v_time_true)
        self.assertTrue(abs(np.sum(v_units_true - v_units)) < 1e-16)
        self.assertTrue(abs(np.sum(v_time_true - v_time)) < self.N * 1e-1)
        self.assertTrue(abs(v_units_time_true - v_units_time) < 1e-1)

    def test_compound_variance(self):
        v = np.array([np.random.normal(self.v_mu, np.sqrt(i + 1), self.T)
                     for i in range(self.N)])
        v_compound = ctana.compound_variance(v)
        v_compound_true = np.var(np.sum(v, axis=0))
        self.assertTrue(abs(v_compound_true - v_compound) < 1e-15)

    def test_powerspec(self):
        sp = cthlp.create_poisson_spiketrains(self.rate, self.T, self.Neff)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0, self.N)
        bins, bsp = cthlp.instantaneous_spike_count(sp_srt, self.tbin)
        freq, power = ctana.powerspec(bsp, self.tbin)
        for i in range(self.N):
            # power(0) == 1./T*integral(s(t))**2 == 1./T*sum(s_binned)**2
            self.assertTrue(
                abs(power[i][0] - 1. / self.T * 1e3
                    * (np.sum(bsp[i])) ** 2) < 1e-15)
        bsp = cthlp.centralize(bsp, time=True)
        freq, power = ctana.powerspec(bsp, self.tbin)
        for i in range(self.N):
            # power(0) == 0
            self.assertTrue(abs(power[i][0]) < 1e-15)

        auto = np.array([np.fft.ifft(x) for x in power])
        for i in range(self.N):
            if np.sum(power[i]) > 0.:
                # power == rate (flat spectrum for poisson with power == rate)
                self.assertTrue(
                    abs(np.mean(power[i]) - self.rate) < self.rate * 2e-1)
                # auto(t) = rate*delta(t)-(offset due to centralizing)
                self.assertTrue(abs(auto[i][0] - self.rate) < self.rate * 2e-1)
                # integral(auto(t)) == 0 (due to centralizing, delta is
                # canceled by offset)
                self.assertTrue(abs(np.sum(auto[i])) < 1e-11)

        freq, power = ctana.powerspec(bsp, self.tbin, Df=self.Df)
        # smallest frequency is larger than size of smoothing window
        self.assertTrue(self.Df <= freq[1])

        freq_units, power_units = ctana.powerspec(
            bsp, self.tbin, Df=self.Df, units=True)
        # power_units should equal population averaged power spectrum
        self.assertTrue(
            abs(np.sum(power_units - np.mean(power, axis=0))) < 1e-10)

    def test_compound_powerspec(self):
        sp = cthlp.create_poisson_spiketrains(self.rate, self.T, self.Neff)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0, self.N)
        bins, bsp = cthlp.instantaneous_spike_count(sp_srt, self.tbin)
        bsp = cthlp.centralize(bsp, time=True)
        freq_alt, power_alt = ctana.powerspec(
            [np.sum(bsp, axis=0)], self.tbin, Df=self.Df)
        freq, power = ctana.compound_powerspec(bsp, self.tbin, Df=self.Df)
        self.assertEqual(len(freq_alt), len(freq))  # frequencies
        self.assertEqual(len(power_alt[0]), len(power))  # same number of bins
        # same spectra
        self.assertTrue(abs(np.sum(power_alt[0] - power)) < 1e-16)

    def test_crossspec(self):
        # use less neurons (0.2*self.N) for full matrix (memory error!)
        Nloc = int(0.2 * self.N)
        Nloceff = int(0.2 * self.Neff)
        sp = cthlp.create_poisson_spiketrains(self.rate, self.T, Nloceff)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0, Nloc)
        bins, bsp = cthlp.instantaneous_spike_count(sp_srt, self.tbin)
        bsp = cthlp.centralize(bsp, time=True)
        freq_power, power = ctana.powerspec(bsp, self.tbin)
        freq_cross, cross = ctana.crossspec(bsp, self.tbin)
        self.assertEqual(len(freq_power), len(freq_cross))
        self.assertEqual(np.min(freq_power), np.min(freq_cross))
        self.assertEqual(np.max(freq_power), np.max(freq_cross))
        for i in range(Nloc):
            for j in range(Nloc):
                if i != j:
                    # poisson trains are uncorrelated
                    self.assertTrue(abs(np.mean(cross[i, j])) < 1e0)
                else:
                    # compare with auto spectra
                    self.assertTrue(
                        abs(np.mean(cross[i, i] - power[i])) < 1e-12)
        sp = cthlp.create_poisson_spiketrains(self.rate, self.T, self.N)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp)
        bins, bsp = cthlp.instantaneous_spike_count(sp_srt, self.tbin)
        bsp = cthlp.centralize(bsp, time=True)
        freq_cross, cross = ctana.crossspec(bsp, self.tbin, units=True)
        self.assertTrue(abs(np.mean(cross)) < 1e-2)
        freq_cross, cross = ctana.crossspec(
            bsp, self.tbin, Df=self.Df, units=True)
        self.assertTrue(self.Df <= freq_cross[1])

    def test_compound_crossspec(self):
        # use less neurons (0.2*self.N) for full matrix (memory error!)
        Nloc = int(0.2 * self.N)
        Nloceff = int(0.2 * self.Neff)
        # population a
        sp_a = cthlp.create_poisson_spiketrains(self.rate, self.T, Nloceff)
        sp_a_ids, sp_a_srt = cthlp.sort_gdf_by_id(sp_a, 0, Nloc)
        bins_a, bsp_a = cthlp.instantaneous_spike_count(
            sp_a_srt, self.tbin, tmin=0., tmax=self.T)
        bsp_a = cthlp.centralize(bsp_a, time=True)
        # population b
        sp_b = cthlp.create_poisson_spiketrains(self.rate, self.T, Nloceff)
        sp_b_ids, sp_b_srt = cthlp.sort_gdf_by_id(sp_b, 0, Nloc)
        bins_b, bsp_b = cthlp.instantaneous_spike_count(
            sp_b_srt, self.tbin, tmin=0., tmax=self.T)
        bsp_b = cthlp.centralize(bsp_b, time=True)
        freq_a, power_a = ctana.compound_powerspec(bsp_a, self.tbin)
        freq_b, power_b = ctana.compound_powerspec(bsp_b, self.tbin)
        freq_cross, cross = ctana.compound_crossspec([bsp_a, bsp_b], self.tbin)
        self.assertTrue(abs(np.sum(power_a - cross[0, 0])) < 1e-10)
        self.assertTrue(abs(np.sum(power_b - cross[1, 1])) < 1e-10)
        freq_cross_alt, cross_alt = ctana.crossspec(
            np.array([np.sum(bsp_a, axis=0), np.sum(bsp_b, axis=0)]),
            self.tbin)
        self.assertTrue(abs(np.sum(cross_alt[0, 1] - cross[0, 1])) < 1e-12)
        self.assertTrue(abs(np.sum(cross_alt[1, 0] - cross[1, 0])) < 1e-12)

    def test_coherence(self):
        # use less neurons (0.2*self.N) for full matrix (memory error!)
        Nloc = int(0.15 * self.N)
        Nloceff = int(0.15 * self.Neff)
        sp = cthlp.create_correlated_spiketrains_sip(
            self.rate, self.T, Nloceff, self.cc)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0, Nloc)
        bins, bsp = cthlp.instantaneous_spike_count(sp_srt, self.tbin)
        # test all pairs of active neurons
        # TODO why does the result not depend on centralizing?
        # bsp = cthlp.centralize(bsp, time=True)
        freq_cross, cross = ctana.crossspec(bsp, self.tbin, Df=self.Df)
        df = freq_cross[1] - freq_cross[0]
        a_lfcoh = []
        for i in range(Nloc):
            for j in range(Nloc):
                if i != j:
                    if np.sum(cross[i, i]) > 0. and np.sum(cross[j, j]) > 0.:
                        lfcoh = np.mean(
                            (np.real(cross[i, j]) /
                             np.sqrt(cross[i, i] * cross[j, j]))
                            [:int(self.fcut / df)])
                        a_lfcoh.append(lfcoh)
                        self.assertTrue(abs(lfcoh - self.cc) < self.cc * 4e-1)
                    else:
                        a_lfcoh.append(0.)
        # average correlation coefficient is p**2 smaller than correlation
        # between active neurons
        self.assertTrue(abs(np.mean(a_lfcoh) - self.p ** 2 * self.cc)
                        < self.cc * 1e-1)
        # test coherence of population averaged signals
        # (careful with interpretation!)
        freq_power, power = ctana.powerspec(
            bsp, self.tbin, Df=self.Df, units=True)
        freq_cross, cross = ctana.crossspec(
            bsp, self.tbin, Df=self.Df, units=True)
        # make sure frequencies are the same for power and cross
        self.assertEqual(len(freq_power), len(freq_cross))
        self.assertEqual(np.min(freq_power), np.min(freq_cross))
        self.assertEqual(np.max(freq_power), np.max(freq_cross))
        df = freq_cross[1] - freq_cross[0]
        lfcoh = np.mean((cross / power)[:int(self.fcut / df)])
        # low frequency coherence should coincide with corrcoef
        self.assertTrue(abs(lfcoh - self.p * self.cc) < self.cc * 1e-1)

    # TODO test structure for gamma process
    def test_autocorrfunc(self):
        Nloc = int(0.1 * self.N)
        Nloceff = int(0.1 * self.Neff)
        # sp = cthlp.create_correlated_spiketrains_sip(
        # self.rate,self.T,Nloceff,self.cc)
        sp = cthlp.create_poisson_spiketrains(self.rate, self.T, Nloceff)
        # sp = cthlp.create_gamma_spiketrains(self.rate,self.T,Nloceff,.5)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0, Nloc)
        # sp_srt = [np.arange(0,self.T,100.) for i in xrange(Nloceff)]
        bins, bsp = cthlp.instantaneous_spike_count(sp_srt, self.tbin)

        freq, power = ctana.powerspec(bsp, self.tbin)
        # freq_cross,cross = ctana.crossspec(bsp,self.tbin,units=True)
        time_auto, autof = ctana.autocorrfunc(freq, power)
        # time_cross,crossf = ctana.crosscorrfunc(freq_cross,cross)
        for i in range(Nloceff):
            if len(autof[i]) % 2 == 0:
                mid = len(autof[i]) / 2 - 1
            else:
                mid = np.floor(len(autof[i]) / 2.)
            offset = self.tbin / self.T * \
                (self.rate + self.rate ** 2 * self.T * 1e-3)
            # a(0) == rate+offset
            self.assertTrue(abs(autof[i][mid] - (self.rate + offset))
                            < (self.rate + offset) * 2e-1)
            # test offset (see notes)
            self.assertTrue(
                abs(np.mean(autof[i][:mid - 1]) - offset) < offset * 2e-1)
        freq, power = ctana.powerspec(bsp, self.tbin, units=True)
        time_auto, autof = ctana.autocorrfunc(freq, power)
        print(autof)
        if len(autof) % 2 == 0:
            mid = len(autof) / 2 - 1
        else:
            mid = np.floor(len(autof) / 2.)
        offset = self.p * self.tbin / self.T * \
            (self.rate + self.rate ** 2 * self.T * 1e-3)
        # mean(a(0)) == p*rate+offset
        self.assertTrue(abs(autof[mid] - (self.p * self.rate + offset))
                        < (self.p * self.rate + offset) * 1e-1)
        # test offset (see notes)
        self.assertTrue(abs(np.mean(autof[:mid - 1]) - offset) < offset * 2e-1)
        # symmetry of autocorrelation function
        lim = np.floor(len(autof) / 4)
        print(autof[mid - lim + 1:mid], len(autof[mid - lim + 1:mid]))
        print((autof[mid + 1:mid + lim])[::-1], len(autof[mid + 1:mid + lim][::-1]))
        print(abs(np.sum(autof[mid - lim + 1:mid] -
                                   (autof[mid + 1:mid + lim])[::-1])))
        self.assertTrue(abs(np.sum(autof[mid - lim + 1:mid] -
                                   (autof[mid + 1:mid + lim])[::-1])) < 1e-12)

    def test_autocorrfunc_time(self):
        Nloc = int(0.1 * self.N)
        Nloceff = int(0.1 * self.Neff)
        sp = cthlp.create_poisson_spiketrains(self.rate, self.T, Nloceff)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0, Nloc)
        time_auto, autof = ctana.autocorrfunc_time(sp_srt, self.tau_max, self.tbin, self.T)
        for i in range(Nloceff):
            if len(autof[i]) % 2 == 0:
                mid = len(autof[i]) / 2 - 1
            else:
                mid = np.floor(len(autof[i]) / 2.)
            offset = self.tbin / self.T * \
                (self.rate + self.rate ** 2 * self.T * 1e-3)
            # a(0) == rate+offset
            self.assertTrue(abs(autof[i][mid] - (self.rate + offset))
                            < (self.rate + offset) * 2e-1)
            # test offset (see notes)
            self.assertTrue(
                abs(np.mean(autof[i][:mid - 1]) - offset) < offset * 2e-1)
        lim = len(autof) / 4
        time_auto, autof = ctana.autocorrfunc_time(sp_srt, self.tau_max, self.tbin, self.T, units=True)
        if len(autof) % 2 == 0:
            mid = len(autof) / 2 - 1
        else:
            mid = np.floor(len(autof) / 2.)
        offset = self.p * self.tbin / self.T * \
            (self.rate + self.rate ** 2 * self.T * 1e-3)
        # mean(a(0)) == p*rate+offset
        self.assertTrue(abs(autof[mid] - (self.p * self.rate + offset))
                        < (self.p * self.rate + offset) * 1e-1)
        # test offset (see notes)
        self.assertTrue(abs(np.mean(autof[:mid - 1]) - offset) < offset * 2e-1)
        # symmetry of autocorrelation function
        lim = len(autof) / 4
        sum_left = np.sum(autof[mid - lim + 1:mid])
        sum_right = np.sum(autof[mid + 1:mid + lim])
        sum_av = (sum_left + sum_right)/2.
        self.assertTrue(abs(sum_left-sum_right)/sum_av < 1e-3)
        
    def test_crosscorrfunc(self):
        Nloc = int(0.1 * self.N)
        Nloceff = int(0.1 * self.Neff)
        sp = cthlp.create_correlated_spiketrains_sip(
            self.rate, self.T, Nloceff, self.cc)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0, Nloc)
        bins, bsp = cthlp.instantaneous_spike_count(sp_srt, self.tbin)

        freq, power = ctana.powerspec(bsp, self.tbin)
        freq_cross, cross = ctana.crossspec(bsp, self.tbin)
        time_auto, autof = ctana.autocorrfunc(freq, power)
        time_cross, crossf = ctana.crosscorrfunc(freq_cross, cross)

        if len(crossf[0, 0]) % 2 == 0:
            mid = len(crossf[0, 0]) / 2 - 1
        else:
            mid = np.floor(len(crossf[0, 0]) / 2.)
        offset = self.tbin / self.T * \
            (self.rate + self.rate ** 2 * self.T * 1e-3)
        for i in range(Nloceff):
            # consistency check with auto-correlation function
            self.assertTrue(abs(np.sum(autof[i] - crossf[i, i])) < 1e-10)
            for j in range(Nloceff):
                if i != j:
                    # c(0) = corrcoef*rate+offset
                    self.assertTrue(
                        abs(crossf[i, j][mid] - (self.cc * self.rate + offset))
                        < (self.cc * self.rate + offset) * 1e-1)
                    # c(0)/a(0) = corrcoef
                    self.assertTrue(
                        abs((crossf[i, j][mid] - offset) /
                            np.sqrt((crossf[i, i][mid] - offset)
                                    * (crossf[j, j][mid] - offset)) - self.cc)
                        < self.cc * 5e-2)

        freq, power = ctana.powerspec(bsp, self.tbin, units=True)
        freq_cross, cross = ctana.crossspec(bsp, self.tbin, units=True)
        time, autof = ctana.autocorrfunc(freq, power)
        time_cross, crossf = ctana.crosscorrfunc(freq, cross)
        offset_auto = self.p * self.tbin / self.T * \
            (self.rate + self.rate ** 2 * self.T * 1e-3)
        offset_cross = 1. * Nloceff * \
            (Nloceff - 1) / Nloc / (Nloc - 1) * self.tbin / \
            self.T * (self.rate + self.rate ** 2 * self.T * 1e-3)
        # c(0) ~ self.p**2*corrcoef*rate+offset
        self.assertTrue(
            abs(crossf[mid] -
                (1. * Nloceff * (Nloceff - 1) / Nloc / (Nloc - 1)
                 * self.cc * self.rate + offset_cross))
            < (1. * Nloceff * (Nloceff - 1) / Nloc / (Nloc - 1)
               * self.cc * self.rate + offset_cross) * 2e-1)

        # c(0)/a(0) = corrcoef
        self.assertTrue(abs((crossf[mid] - offset_cross)
                            / (autof[mid] - offset_auto)
                            - 1. * (Nloceff - 1.) / (Nloc - 1.) * self.cc)
                        < 1. * (Nloceff - 1.) / (Nloc - 1.) * self.cc * 2e-1)

    def test_corrcoef(self):
        Nloc = int(0.1 * self.N)
        Nloceff = int(0.1 * self.Neff)
        sp = cthlp.create_correlated_spiketrains_sip(
            self.rate, self.T, Nloceff, self.cc)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0, Nloc)
        bins, bsp = cthlp.instantaneous_spike_count(sp_srt, self.tbin)

        freq_cross, cross = ctana.crossspec(bsp, self.tbin)
        time_cross, crossf = ctana.crosscorrfunc(freq_cross, cross)

        corrcoef = ctana.corrcoef(time_cross, crossf)
        for i in range(Nloc):
            for j in range(Nloc):
                if i < Nloceff and j < Nloceff:
                    if i == j:
                        self.assertTrue(abs(corrcoef[i, j] - 1.) < 1e-12)
                    else:
                        self.assertTrue(
                            abs(corrcoef[i, j] - self.cc) < self.cc * 1e-1)
                else:
                    self.assertTrue(abs(corrcoef[i, j]) < 1e-12)

if __name__ == '__main__':
    unittest.main()
