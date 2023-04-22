#!/usr/bin/env python3

"""
  Analysis class to read a ROOT TTree of MC track information
  and do jet-finding, and save response histograms.
  
  Author: Ezra Lesser (elesser@berkeley.edu)
          based on substructure framework by James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import os
import sys
import argparse

# Data analysis and plotting
import numpy as np
import ROOT
import array
import math

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjext
import fjcontrib
import ecorrel

# Analysis utilities
from pyjetty.alice_analysis.process.user.substructure import process_parton_hadron_base

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

################################################################
class ProcessPH_ENC(process_parton_hadron_base.ProcessPHBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):

    # Initialize base class
    super(ProcessPH_ENC, self).__init__(
      input_file, config_file, output_dir, debug_level, **kwargs)

    self.obs_name = "R_{L}"

  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects_R(self, jetR):

    for observable in self.observable_list:

      for i, trk_thrd in enumerate(self.obs_settings[observable]):

        obs_setting = trk_thrd
        grooming_setting = self.obs_grooming_settings[observable][i]
        obs_label = self.utils.obs_label(obs_setting, grooming_setting) # NB: used to be obs_label = self.utils.obs_label(trk_thrd, None) 

        # Init ENC histograms
        for (level_1, level_2, MPI) in self.RM_levels:
          # 2D histograms for matched jet pt correlations (e.g full jet pt vs ch jet pt)
          if 'jet_pt' in observable:
              name = 'h_matched_{}_{}_{}_MPI{}_R{}_{}'.format(observable, level_1, level_2, MPI, jetR, obs_label) 
              pt_bins = linbins(0,200,200)
              h = ROOT.TH2D(name, name, 200, pt_bins, 200, pt_bins)
              h.GetXaxis().SetTitle('pT ({} jet)'.format(level_1)) # FIX ME: find a better way to format
              h.GetYaxis().SetTitle('pT ({} jet)'.format(level_2))
              setattr(self, name, h)

          for level in [level_1, level_2]:
            # 2D histograms of ENC vs jet pt (for matched jets)
            # NB: there shouldn't be any efficiency problem hence using the matched or unmatched jets should give the same resutls?
            if 'ENC' in observable:
              for ipoint in range(2, 3):
                name = 'h_{}_JetPt_{}_MPI{}_R{}_{}'.format(observable + str(ipoint), level, MPI, jetR, obs_label)
                print('Initialize histogram',name)
                pt_bins = linbins(0,200,200)
                RL_bins = logbins(1E-4,1,50)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                h.GetXaxis().SetTitle('pT ({} jet)'.format(level)) # FIX ME: double-check about formatting
                h.GetYaxis().SetTitle('R_{L}')
                setattr(self, name, h)

                name = 'h_matched_{}_JetPt_{}_MPI{}_R{}_{}'.format(observable + str(ipoint), level, MPI, jetR, obs_label)
                pt_bins = linbins(0,200,200)
                RL_bins = logbins(1E-4,1,50)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                h.GetXaxis().SetTitle('pT ({} jet)'.format(level)) # FIX ME: double-check about formatting
                h.GetYaxis().SetTitle('R_{L}')
                setattr(self, name, h)

            # 1D histograms of jet pt (for matched jets)
            if 'jet_pt' in observable:
              name = 'h_{}_JetPt_{}_MPI{}_R{}_{}'.format(observable, level, MPI, jetR, obs_label)
              pt_bins = linbins(0,200,200)
              h = ROOT.TH1D(name, name, 200, pt_bins)
              h.GetXaxis().SetTitle('pT ({} jet)'.format(level))
              h.GetYaxis().SetTitle('Counts')
              setattr(self, name, h)

              name = 'h_matched_{}_JetPt_{}_MPI{}_R{}_{}'.format(observable, level, MPI, jetR, obs_label)
              pt_bins = linbins(0,200,200)
              h = ROOT.TH1D(name, name, 200, pt_bins)
              h.GetXaxis().SetTitle('pT ({} jet)'.format(level))
              h.GetYaxis().SetTitle('Counts')
              setattr(self, name, h)

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # Fill 2D histogram of (pt, obs)
  #---------------------------------------------------------------
  def fill_observable_histograms(self, jet, jet_groomed_lund, jetR, level, MPI,
      obs_setting, grooming_setting, obs_label, jet_pt_ungroomed):

    # Only doing the MPI scaling at ch level, so ignore everything else
    # FIX ME: do I really want to keep this? -- comment it out for now
    # if level != "ch":
    #   return
    if level == "p":
      return

    # Calculate observables and fill histograms
    constituents = fj.sorted_by_pt(jet.constituents())
    c_select = fj.vectorPJ()
    trk_thrd = obs_setting # same info as obs_label (i.e. track threshold value in this ENC analysis)

    for c in constituents:
      if c.pt() < trk_thrd:
        break
      c_select.append(c) # NB: use the break statement since constituents are already sorted
    
    dphi_cut = -9999
    deta_cut = -9999
    new_corr = ecorrel.CorrelatorBuilder(c_select, jet.perp(), 2, 1, dphi_cut, deta_cut)
    for observable in self.observable_list:
      if 'ENC' in observable:
        for ipoint in range(2, 3):
          for index in range(new_corr.correlator(ipoint).rs().size()):
            getattr(self, 'h_{}_JetPt_{}_MPI{}_R{}_{}'.format(observable + str(ipoint), level, MPI, jetR, obs_label)).Fill(jet_pt_ungroomed, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])
      if 'jet_pt' in observable:
        getattr(self, 'h_{}_JetPt_{}_MPI{}_R{}_{}'.format(observable, level, MPI, jetR, obs_label)).Fill(jet_pt_ungroomed)

  #---------------------------------------------------------------
  # This function is called per observable per jet subconfigration 
  # used in fill_matched_jet_histograms
  # This function is created because we cannot use fill_observable_histograms 
  # directly because observable list loop inside that function
  #---------------------------------------------------------------
  def fill_matched_observable_histograms(self, jet, jet_groomed_lund, jetR, level, MPI,
      obs_setting, grooming_setting, obs_label, jet_pt_ungroomed, suffix):
    
    constituents = fj.sorted_by_pt(jet.constituents())
    c_select = fj.vectorPJ()
    trk_thrd = obs_setting

    for c in constituents:
      if c.pt() < trk_thrd:
        break
      c_select.append(c) # NB: use the break statement since constituents are already sorted
    
    dphi_cut = -9999
    deta_cut = -9999
    new_corr = ecorrel.CorrelatorBuilder(c_select, jet.perp(), 2, 1, dphi_cut, deta_cut)
    for observable in self.observable_list:
      if 'ENC' in observable:
        for ipoint in range(2, 3):
          for index in range(new_corr.correlator(ipoint).rs().size()):
            getattr(self, 'h_matched_{}_JetPt_{}_{}_{}'.format(observable + str(ipoint), level, suffix, obs_label)).Fill(jet_pt_ungroomed, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])
      if 'jet_pt' in observable:
        getattr(self, 'h_matched_{}_JetPt_{}_{}_{}'.format(observable, level, suffix, obs_label)).Fill(jet_pt_ungroomed)

  #---------------------------------------------------------------
  # Fill matched jet histograms
  # suffix contains MPI and jet R information
  #---------------------------------------------------------------
  def fill_matched_jet_histograms(self, jetR, obs_setting, grooming_setting, obs_label,
      jet_p, jet_p_groomed_lund, jet_h, jet_h_groomed_lund, jet_ch, jet_ch_groomed_lund,
      jet_pt_p_ungroomed, jet_pt_h_ungroomed, jet_pt_ch_ungroomed, suffix):

    # Compute observables
    obs = None

    pt = { "p" : jet_pt_p_ungroomed,
           "h" : jet_pt_h_ungroomed,
           "ch": jet_pt_ch_ungroomed 
    }

    # print('debug--jet R',jetR,'obs setting',obs_setting)
    # print('debug--charged jets pt',jet_pt_ch_ungroomed,pt["ch"])
    # print('debug--full jets pt',jet_pt_h_ungroomed,pt["h"])

    # fill matched jets
    for (level_1, level_2, MPI) in self.RM_levels:  
      # fill observable histograms
      # loop over observables inside fill_matched_observable_histograms()
      if MPI == "on":
        pass
      else:
        for level in [level_1, level_2]:
          if level == "p":
            self.fill_matched_observable_histograms(jet_p, jet_p_groomed_lund, jetR, level, MPI, obs_setting, grooming_setting, obs_label, pt[level], suffix)
          if level == "h":
            self.fill_matched_observable_histograms(jet_h, jet_h_groomed_lund, jetR, level, MPI, obs_setting, grooming_setting, obs_label, pt[level], suffix)
          if level == "ch":
            self.fill_matched_observable_histograms(jet_ch, jet_ch_groomed_lund, jetR, level, MPI, obs_setting, grooming_setting, obs_label, pt["h"], suffix)

        # This 2D correlation histograms need to be filled outside fill_matched_observable_histograms()
        for observable in self.observable_list:
          if 'jet_pt' in observable:
            hname = 'h_matched_{}_{}_{}_{}_{}'.format(observable, level_1, level_2, suffix, obs_label) # FIX ME: this is a 2D histogram which is not defined yet
            getattr(self, hname).Fill(pt[level_1], pt[level_2]) # Fill matched jet 2D histograms (e.g. full jet vs ch jet)


##################################################################
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Process MC')
  parser.add_argument('-f', '--inputFile', action='store',
                      type=str, metavar='inputFile',
                      default='AnalysisResults.root',
                      help='Path of ROOT file containing TTrees')
  parser.add_argument('-c', '--configFile', action='store',
                      type=str, metavar='configFile',
                      default='config/analysis_config.yaml',
                      help="Path of config file for analysis")
  parser.add_argument('-o', '--outputDir', action='store',
                      type=str, metavar='outputDir',
                      default='./TestOutput',
                      help='Output directory for output to be written to')

  # Parse the arguments
  args = parser.parse_args()

  print('Configuring...')
  print('inputFile: \'{0}\''.format(args.inputFile))
  print('configFile: \'{0}\''.format(args.configFile))
  print('ouputDir: \'{0}\"'.format(args.outputDir))

  # If invalid inputFile is given, exit
  if not os.path.exists(args.inputFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.inputFile))
    sys.exit(0)

  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  analysis = ProcessPH_ENC(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_mc()