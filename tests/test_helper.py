# global imports
import unittest
import numpy as np

# local imports
import correlation_toolbox.helper as cthlp


class TestHelper(unittest.TestCase):

    def setUp(self):
        np.random.seed(12345)
        self.rate = 30.  # (Hz)
        self.T = 3e4  # (ms)
        self.N = 100
        self.p = 0.6  # percentage of neurons active
        self.Neff = int(self.p * self.N)
        self.cc = 0.3
        self.tbin = 1.  # (ms)

    def test_create_poisson_spiketrains(self):
        sp = cthlp.create_poisson_spiketrains(self.rate, self.T, self.N)
        self.assertEqual(self.N, len(np.unique(sp[:, 0])))  # N
        self.assertTrue(self.T >= np.max(sp[:, 1]))  # T
        emp_rate = 1. * len(sp) / self.T * 1e3 / self.N
        self.assertTrue(abs(self.rate - emp_rate) < 1e0)  # rate

    def test_sort_gdf_by_id(self):
        # create N-5 poisson instead of N, creates empty arrays in sp_srt
        sp = cthlp.create_poisson_spiketrains(self.rate, self.T, self.Neff)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0, self.N)
        self.assertEqual(self.N, len(sp_ids))  # N
        self.assertTrue(self.T >= np.max([np.max(x)
                        for x in sp_srt if len(x) > 0]))  # T
        for i in range(self.N):
            emp_rate = 1. * len(sp_srt[i]) / self.T * 1e3
            assert(emp_rate >= 0.)
            if emp_rate > 0.:
                self.assertTrue(abs(self.rate - emp_rate) < 1e1)  # rate
                self.assertTrue(min(np.diff(sp_srt[i])) > 0.)  # time ordering

    def test_instantaneous_spike_count(self):
        # create N-5 poisson instead of N, creates empty arrays in sp_srt
        # to test binning for empty spiketrains
        sp = cthlp.create_poisson_spiketrains(self.rate, self.T, self.Neff)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0, self.N)
        bins, bsp = cthlp.instantaneous_spike_count(sp_srt, self.tbin)

        # test whether binning produces correct results
        sp_srt = np.array([[1., 2., 5., 7.], [4., 6., 9.]])
        # ground truth
        bsp_true = np.array(
            [[1, 1, 0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1, 0, 1]])
        bins, bsp = cthlp.instantaneous_spike_count(sp_srt, self.tbin)
        self.assertTrue(len(bins) == len(bsp[0]))  # number of bins
        self.assertEqual(2, len(bsp))  # number of binned spike trains
        self.assertEqual(np.sum(bsp_true - bsp), 0.)  # histogram

    def test_create_correlated_spiketrains_sip(self):
        # create N-5 poisson instead of N, changes correlation
        sp = cthlp.create_correlated_spiketrains_sip(
            self.rate, self.T, self.Neff, self.cc)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0, self.N)
        bins, bsp = cthlp.instantaneous_spike_count(sp_srt, self.tbin)
        emp_rate = 1. * np.sum(bsp) / self.T * 1e3 / self.N
        self.assertTrue(abs(self.p * self.rate - emp_rate) < 5e-1)  # rate
        self.assertEqual(self.N, len(bsp))  # N
        self.assertTrue(self.T >= np.max(bins))  # T
        emp_cc = np.corrcoef(cthlp.strip_binned_spiketrains(bsp))
        emp_a_cc = []
        for i in range(self.Neff):
            for j in range(self.Neff):
                if i != j:
                    emp_a_cc.append(emp_cc[i, j])
        emp_mu_cc = 1. / (self.N * (self.N - 1.)) * np.sum(emp_a_cc)
        # correlation coefficient
        self.assertTrue(abs(self.p ** 2 * self.cc - emp_mu_cc) < 2e-2)

    def test_centralize(self):
        v1 = np.random.normal(-50, 2, self.T * 1e1)
        v2 = np.random.normal(-30, 2, self.T * 1e1)
        v_cen_time = cthlp.centralize([v1, v2], time=True)
        for v in v_cen_time:
            self.assertTrue(abs(np.mean(v)) < 1e-12)
        v_cen_units = cthlp.centralize([v1, v2], units=True)
        for v in v_cen_units.T:
            self.assertTrue(abs(np.mean(v)) < 1e-12)
        v_cen_timeunits = cthlp.centralize([v1, v2], time=True, units=True)
        self.assertTrue(abs(np.mean(v_cen_timeunits)) < 1e-12)

    def test_strip_sorted_spiketrains(self):
        sp = cthlp.create_poisson_spiketrains(self.rate, self.T, self.Neff)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0., self.N)
        self.assertEqual(self.N, len(sp_srt))
        sp_srt = cthlp.strip_sorted_spiketrains(sp_srt)
        self.assertEqual(self.Neff, len(sp_srt))

    def test_strip_binned_spiketrains(self):
        sp = cthlp.create_poisson_spiketrains(self.rate, self.T, self.Neff)
        sp_ids, sp_srt = cthlp.sort_gdf_by_id(sp, 0., self.N)
        bins, bsp = cthlp.instantaneous_spike_count(sp_srt, self.tbin)
        self.assertEqual(self.N, len(bsp))
        bsp = cthlp.strip_binned_spiketrains(bsp)
        self.assertEqual(self.Neff, len(bsp))

if __name__ == '__main__':
    unittest.main()
