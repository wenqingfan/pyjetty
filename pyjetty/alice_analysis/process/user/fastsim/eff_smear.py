#!/usr/bin/env python3

'''
Loads truth-level MC track information and applies effciency cuts and pT smearing
to emulate a fast simulation, and saves resulting ROOT files.

Written by Ezra Lesser (elesser@berkeley.edu), Spring 2020
Some code taken from processing script by James Mulligan
'''

from __future__ import print_function, division

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import math
import random
import ROOT

import fastjet as fj
import fjext

import ecorrel

from pyjetty.alice_analysis.process.base import process_io
from pyjetty.alice_analysis.process.base import process_io_gen
from pyjetty.cstoy import alice_efficiency


# Approximated fit for MC calculated (LHC18b8) pT resolution
def sigma_pt(pt):
    if pt < 1:
        return pt * (-0.035 * pt + 0.04)
    elif pt < 60:
        return pt * (0.00085 * pt + 0.00415)
    # else: approximately valid for at least pt < 90 GeV
    return pt * (0.0015 * pt - 0.035)

#################################################################################
class eff_smear:

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, inputFile='', outputDir='', is_jetscape=False):
        self.input_file = inputFile
        self.output_dir = outputDir
        self.is_jetscape = is_jetscape
        self.pair_eff_file = ROOT.TFile.Open("PairEff.root","READ")
        self.dpbin = 4;
        self.dp_lo = [0, 0.2, 0.4, 1]
        self.dp_hi = [0.2, 0.4, 1, 2]
        self.h1d_eff_vs_dR_in_dq_over_p = []
        for idp in range(self.dpbin):
            hname = 'h1d_eff_vs_dR_in_dq_over_p_{}'.format(idp)
            self.h1d_eff_vs_dR_in_dq_over_p.append( ROOT.TH1D(self.pair_eff_file.Get(hname)) )

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def eff_smear(self):

        start_time = time.time()

        # ------------------------------------------------------------------------

        # Initialize dataframes from data
        self.init_df()
        print('--- {} seconds ---'.format(time.time() - start_time))

        # ------------------------------------------------------------------------

        # Build truth-level histogram of track pT multiplicity
        print("Building truth-level track pT histogram...")
        self.hist_list.append( ("truth_pt", self.build_pt_hist()) )
        print('--- {} seconds ---'.format(time.time() - start_time))

        print("Building truth-level track pid histogram...")
        self.hist_list.append( ("truth_pid", self.build_pid_hist()) )
        print('--- {} seconds ---'.format(time.time() - start_time))

        # Apply eta cut at the end of the TPC
        self.df_fjparticles = self.apply_eta_cut(self.df_fjparticles)
        print('--- {} seconds ---'.format(time.time() - start_time))

        # Apply efficiency cut
        self.df_fjparticles = self.apply_eff_cut(self.df_fjparticles)
        print('--- {} seconds ---'.format(time.time() - start_time))

        # Apply pair efficiency cut
        self.df_fjparticles = self.apply_pair_eff_cut(self.df_fjparticles)
        print('--- {} seconds ---'.format(time.time() - start_time))

        # Build truth-level histogram of track pT multiplicity after efficiency cuts
        print("Building truth-level track pT histogram after efficiency cuts...")
        self.hist_list.append( ("truth_pt_eff_cuts", self.build_pt_hist()) )
        print('--- {} seconds ---'.format(time.time() - start_time))

        # Apply pT smearing
        self.df_fjparticles = self.apply_pt_smear(self.df_fjparticles)
        print('--- {} seconds ---'.format(time.time() - start_time))

        # Build truth-level histogram of track pT multiplicity
        print("Building detector-level track pT histogram...")
        self.hist_list.append( ("fastsim_pt", self.build_pt_hist()) )
        print('--- {} seconds ---'.format(time.time() - start_time))

        # ------------------------------------------------------------------------

        # Write data to file
        print(self.df_fjparticles)
        print("Writing fast simulation to ROOT TTree...")
        self.io.save_dataframe("AnalysisResultsFastSim.root", self.df_fjparticles,
                               df_true=True, histograms=self.hist_list, is_jetscape=self.is_jetscape)
        print('--- {} seconds ---'.format(time.time() - start_time))


    #---------------------------------------------------------------
    # Initialize dataframe from input data
    #---------------------------------------------------------------
    def init_df(self):
        # Use IO helper class to convert truth-level ROOT TTree into
        # a SeriesGroupBy object of fastjet particles per event
        self.io = process_io_gen.ProcessIO(input_file=self.input_file, output_dir=self.output_dir,
                                        tree_dir='PWGHF_TreeCreator',
                                        track_tree_name='tree_Particle_gen',
                                        use_ev_id_ext=False,
                                        is_jetscape=self.is_jetscape)
        self.df_fjparticles = self.io.load_dataframe()
        self.nTracks_truth = len(self.df_fjparticles)
        print("DataFrame loaded from data.")
        print(self.df_fjparticles)
        print(f'columns: {list(self.df_fjparticles.columns)}')
        # Initialize a list of histograms to be written to file
        self.hist_list = []

        ev_id = self.df_fjparticles['ev_id'].to_numpy()
        ev_id_unique = np.unique(ev_id)
        self.nevents = len(ev_id_unique)
        print("original event size",self.nevents)

    #---------------------------------------------------------------
    # Apply eta cuts
    #---------------------------------------------------------------
    def apply_eta_cut(self, df):
        df = df[df["ParticleEta"].map(abs) < 0.9]
        print("%i out of %i total truth tracks deleted after eta cut." % \
              (self.nTracks_truth - len(df), self.nTracks_truth))
        return df

    #---------------------------------------------------------------
    # Build histogram of pT values and return it
    #---------------------------------------------------------------
    def build_pt_hist(self):
        bins = np.concatenate((np.arange(0, 0.3, 0.05), np.arange(0.3, 1, 0.1), np.arange(1, 3, 0.2), 
                               np.arange(3, 10, 0.5), np.arange(10, 20, 1),
                               np.arange(20, 50, 2), np.arange(50, 155, 5)))
        return np.histogram(self.df_fjparticles["ParticlePt"], bins=bins)

    #---------------------------------------------------------------
    # Build histogram of PID values and return it (FIX ME: can be deleted later)
    #---------------------------------------------------------------
    def build_pid_hist(self):
        bins = np.arange(-500, 500, 1)
        return np.histogram(self.df_fjparticles["ParticlePID"], bins=bins)

    #---------------------------------------------------------------
    # Apply efficiency cuts
    #---------------------------------------------------------------
    def apply_eff_cut(self, df):

        # Apply efficiency cut for fastjet particles
        eff_smearer = alice_efficiency.AliceChargedParticleEfficiency()
        df = df[df["ParticlePt"].map(lambda x: eff_smearer.pass_eff_cut(x))]
        print("%i out of %i total truth tracks deleted after efficiency cut." % \
              (self.nTracks_truth - len(df), self.nTracks_truth))
        return df

    def pass_pair_eff(self, dist, dq_over_p):
        idpbin = -9999
        for idp in range(self.dpbin):
            if math.fabs(dq_over_p)>=self.dp_lo[idp] and math.fabs(dq_over_p)<self.dp_hi[idp]:
                idpbin = idp
        
        pair_eff = 1 # set pair efficeincy to 1 if dq_over_p>=2
        if idpbin>=0:
            if math.log10(dist)<0 and math.log10(dist)>-3:
                ibin = self.h1d_eff_vs_dR_in_dq_over_p[idpbin].FindBin(math.log10(dist))
                pair_eff = self.h1d_eff_vs_dR_in_dq_over_p[idpbin].GetBinContent(ibin)
            elif math.log10(dist)>=0:
                pair_eff = 1 # overflow
            else:
                pair_eff = 0 # underflow
        # print("pair distance",dist,"with eff",pair_eff)
        # pair_eff = 1+0.2*math.log10(dist) # 100% at log(dist)=0 and 40% at log(dist)=-3
        if random.random() < pair_eff:
            return True
        else:
            return False

    #---------------------------------------------------------------
    # Apply pair efficiency cuts
    # 1. Loop through every pair of entries in the data-frame
    # 2. Get pair efficiency for each pair, sample the pairs to be
    #    thrown away. For such pairs, randomly pick one track and
    #    save it to a list
    # 3. Remove the saved list of tracks from the data-frame
    #--------------------------------------------------------------- 
    def apply_pair_eff_cut(self, df):
        # appl some random PID cut
        pt = df['ParticlePt'].to_numpy()
        phi = df['ParticlePhi'].to_numpy()
        eta = df['ParticleEta'].to_numpy()
        ev_id = df['ev_id'].to_numpy()
        ev_id_unique = np.unique(ev_id)
        print("unique event id",ev_id_unique)
        # df = df[df["ev_id"]<2000]

        trk_status_list = np.array([])
        truth_dR_list = np.array([])
        truth_dq_over_p_list = np.array([])
        reco_dR_list = np.array([])
        reco_dq_over_p_list = np.array([])

        for ievt in range(self.nevents):
            if ievt % 100 == 0:
                print("Processing event ",ievt,"/",self.nevents)
            pt_evt = pt[ev_id==ievt]
            phi_evt = phi[ev_id==ievt]
            eta_evt = eta[ev_id==ievt]
            # print("check evt id",ev_id,"pt",pt_evt,"phi",phi_evt,"eta",eta_evt)
            throw_pair_list = np.array([])

            parts_evt = fjext.vectorize_pt_eta_phi(pt_evt, eta_evt, phi_evt)

            new_corr = ecorrel.CorrelatorBuilder(parts_evt, 1.0, 2, 1)

            for index in range(new_corr.correlator(2).rs().size()):
                itrk1 = new_corr.correlator(2).indices1()[index]
                itrk2 = new_corr.correlator(2).indices2()[index]
                
                if itrk1 < itrk2:  # FIX ME: full two loops in the CorrelatorBuilder
                    dist = new_corr.correlator(2).rs()[index]
                    dq_over_p = 1/pt_evt[itrk1]-1/pt_evt[itrk2]

                    truth_dR_list = np.append(truth_dR_list, math.log10(dist))
                    truth_dq_over_p_list = np.append(truth_dq_over_p_list, math.fabs(dq_over_p))
                    if self.pass_pair_eff(dist, dq_over_p)==False:
                            # print("pair not passing pair efficiency check")
                            throw_pair_list = np.append(throw_pair_list, itrk1) # FIX ME: or itrk2
                    else:
                        reco_dR_list = np.append(reco_dR_list, math.log10(dist))
                        reco_dq_over_p_list = np.append(reco_dq_over_p_list, math.fabs(dq_over_p))

            throw_pair_unique_list = np.unique(throw_pair_list)

            # for current events, mark the status for all tracks (True means to be selected)
            for itrk in range(len(pt_evt)):
                if itrk in throw_pair_unique_list:
                    trk_status_list = np.append(trk_status_list,False)
                else:
                    trk_status_list = np.append(trk_status_list,True)

        df['status'] = trk_status_list
        df = df[df["status"]==True]
        print(df)

        df = df.drop('status', axis=1)

        bins = np.arange(-3, 0, 0.03)
        for idp, (dp_lo, dp_hi) in enumerate(zip(self.dp_lo,self.dp_hi)):
            truth_mask = np.where((truth_dq_over_p_list>=dp_lo) & (truth_dq_over_p_list<dp_hi), True, False)
            self.hist_list.append( ('h1d_truth_vs_dR_{}'.format(idp), np.histogram(truth_dR_list[truth_mask], bins=bins)) )
            reco_mask = np.where((reco_dq_over_p_list>=dp_lo) & (reco_dq_over_p_list<dp_hi), True, False)
            self.hist_list.append( ('h1d_reco_vs_dR_{}'.format(idp), np.histogram(reco_dR_list[reco_mask], bins=bins)) )

        return df

    #---------------------------------------------------------------
    # Apply pt smearing
    #---------------------------------------------------------------
    def apply_pt_smear(self, df):
        true_pt = df["ParticlePt"]
        smeared_pt = [ np.random.normal(pt, sigma_pt(pt)) for pt in true_pt ]
        if self.is_jetscape:
            df = pd.DataFrame({"run_number": df["run_number"], "ev_id": df["ev_id"],
                               "ParticlePt": smeared_pt, "ParticleEta": df["ParticleEta"],
                               "ParticlePhi": df["ParticlePhi"], "z_vtx_reco": df["z_vtx_reco"],
                               "is_ev_rej": df["is_ev_rej"],
                               "status": df["status"]})
        else:
            df = pd.DataFrame({"run_number": df["run_number"], "ev_id": df["ev_id"],
                               "ParticlePt": smeared_pt, "ParticleEta": df["ParticleEta"],
                               "ParticlePhi": df["ParticlePhi"], "z_vtx_reco": df["z_vtx_reco"],
                               "is_ev_rej": df["is_ev_rej"]})
        print("pT has been smeared for all tracks.")

        # Create histogram to verify pt smearing distribution
        pt_dif = [ (pt_smear - pt_true) / pt_true for pt_smear, pt_true in zip(smeared_pt, true_pt) ]
        pt_bins = np.concatenate((np.arange(0, 1, 0.1), np.arange(1, 10, .5), np.arange(10, 20, 1),
                                  np.arange(20, 50, 2), np.arange(50, 95, 5)))
        dif_bins = np.arange(-0.5, 0.5, .001)
        pt_smearing_dists = np.histogram2d(true_pt, pt_dif, bins=[pt_bins, dif_bins])
        self.hist_list.append( ("pt_smearing", pt_smearing_dists) )

        return df

##################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fast Simulation Generator")
    parser.add_argument("-i", "--inputFile", action="store", type=str, metavar="inputFile",
                        default="AnalysisResults.root", help="Path of ROOT file containing MC TTrees")
    parser.add_argument("-o", "--outputDir", action="store", type=str, metavar="outputDir",
                        default="./TestOutput", help="Output path for fast sim ROOT TTree")
    parser.add_argument('--jetscape', action='store_true')
    args = parser.parse_args()

    print('Configuring...')
    print('inputFile: \'{0}\''.format(args.inputFile))
    print('ouputDir: \'{0}\"'.format(args.outputDir))
    print(f'is_jetscape: {args.jetscape}')

    # If invalid inputFile is given, exit
    if not os.path.exists(args.inputFile):
        print('File \"{0}\" does not exist! Exiting!'.format(args.inputFile))
        sys.exit(0)

    processor = eff_smear(inputFile=args.inputFile, outputDir=args.outputDir, is_jetscape=args.jetscape)
    processor.eff_smear()
