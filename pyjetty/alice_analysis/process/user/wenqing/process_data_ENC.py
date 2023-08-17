#!/usr/bin/env python3

"""
  Analysis class to read a ROOT TTree of track information
  and do jet-finding, and save basic histograms.
  
  Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import os
import sys
import argparse
import sys

# Data analysis and plotting
import ROOT
import yaml
import numpy as np
import array 
import math
import random

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import ecorrel

# Base class
from pyjetty.alice_analysis.process.user.substructure import process_data_base

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

################################################################
class ProcessData_ENC(process_data_base.ProcessDataBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # print(sys.path)
    # print(sys.modules)

    # Initialize base class
    super(ProcessData_ENC, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    
    self.observable = self.observable_list[0]


  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):

    for jetR in self.jetR_list:
      for observable in self.observable_list:
        for trk_thrd in self.obs_settings[observable]:

          obs_label = self.utils.obs_label(trk_thrd, None) 
          
          if 'ENC' in observable:
            for ipoint in range(2, 3):
              name = 'h_{}_JetPt_R{}_{}'.format(observable + str(ipoint), jetR, trk_thrd)
              pt_bins = linbins(0,200,200)
              RL_bins = logbins(1E-4,1,50)
              h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
              h.GetXaxis().SetTitle('p_{T,ch jet}')
              h.GetYaxis().SetTitle('R_{L}')
              setattr(self, name, h)

              name = 'h_{}Pt_JetPt_R{}_{}'.format(observable + str(ipoint), jetR, trk_thrd)
              pt_bins = linbins(0,200,200)
              ptRL_bins = logbins(1E-3,1E2,60)
              h = ROOT.TH2D(name, name, 200, pt_bins, 60, ptRL_bins)
              h.GetXaxis().SetTitle('p_{T,ch jet}')
              h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
              setattr(self, name, h)

          if 'EEC_noweight' in observable or 'EEC_weight2' in observable:
            name = 'h_{}_JetPt_R{}_{}'.format(observable, jetR, obs_label)
            pt_bins = linbins(0,200,200)
            RL_bins = logbins(1E-4,1,50)
            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
            h.GetXaxis().SetTitle('p_{T,ch jet}')
            h.GetYaxis().SetTitle('R_{L}')
            setattr(self, name, h)

          if 'jet_pt' in observable:
            name = 'h_{}_JetPt_R{}_{}'.format(observable, jetR, trk_thrd)
            pt_bins = linbins(0,200,200)
            h = ROOT.TH1D(name, name, 200, pt_bins)
            h.GetXaxis().SetTitle('p_{T,ch jet}')
            h.GetYaxis().SetTitle('Counts')
            setattr(self, name, h)

            name = 'h_Nconst_JetPt_R{}_{}'.format(jetR, trk_thrd)
            pt_bins = linbins(0,200,200)
            Nconst_bins = linbins(0,50,50)
            h = ROOT.TH2D(name, name, 200, pt_bins, 50, Nconst_bins)
            h.GetXaxis().SetTitle('p_{T,ch jet}')
            h.GetYaxis().SetTitle('N_{const}')
            setattr(self, name, h)

          # fill perp cone histograms
          self.pair_type_labels = ['']
          if self.do_rho_subtraction:
            self.pair_type_labels = ['_bb','_sb','_ss']

          if self.do_perpcone:
            if 'jet_pt' in observable:
              name = 'h_perpcone_{}_JetPt_R{}_{}'.format('pt', jetR, trk_thrd)
              pt_bins = linbins(-200,200,400)
              h = ROOT.TH1D(name, name, 200, pt_bins)
              h.GetXaxis().SetTitle('p_{T,perp cone}')
              h.GetYaxis().SetTitle('Counts')
              setattr(self, name, h)

              name = 'h_perpcone_Nconst_JetPt_R{}_{}'.format(jetR, trk_thrd)
              pt_bins = linbins(-200,200,400)
              Nconst_bins = linbins(0,50,50)
              h = ROOT.TH2D(name, name, 200, pt_bins, 50, Nconst_bins)
              h.GetXaxis().SetTitle('p_{T,ch jet}')
              h.GetYaxis().SetTitle('N_{const}')
              setattr(self, name, h)

            for pair_type_label in self.pair_type_labels:

              if 'ENC' in observable:
                for ipoint in range(2, 3):
                  name = 'h_perpcone_{}_JetPt_R{}_{}'.format(observable + str(ipoint) + pair_type_label, jetR, trk_thrd)
                  pt_bins = linbins(0,200,200)
                  RL_bins = logbins(1E-4,1,50)
                  h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                  h.GetXaxis().SetTitle('p_{T,ch jet}')
                  h.GetYaxis().SetTitle('R_{L}')
                  setattr(self, name, h)

                  name = 'h_perpcone_{}Pt_JetPt_R{}_{}'.format(observable + str(ipoint) + pair_type_label, jetR, trk_thrd)
                  pt_bins = linbins(0,200,200)
                  ptRL_bins = logbins(1E-3,1E2,60)
                  h = ROOT.TH2D(name, name, 200, pt_bins, 60, ptRL_bins)
                  h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
                  setattr(self, name, h)

              if 'EEC_noweight' in observable or 'EEC_weight2' in observable:
                name = 'h_perpcone_{}_JetPt_R{}_{}'.format(observable + pair_type_label, jetR, obs_label)
                pt_bins = linbins(0,200,200)
                RL_bins = logbins(1E-4,1,50)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                h.GetXaxis().SetTitle('p_{T,ch jet}')
                h.GetYaxis().SetTitle('R_{L}')
                setattr(self, name, h)

          if self.do_jetcone:

            if 'ENC' in observable:
              for ipoint in range(2, 3):
                name = 'h_jetcone_{}_JetPt_R{}_{}'.format(observable + str(ipoint), jetR, trk_thrd)
                pt_bins = linbins(0,200,200)
                RL_bins = logbins(1E-4,1,50)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                h.GetXaxis().SetTitle('p_{T,ch jet}')
                h.GetYaxis().SetTitle('R_{L}')
                setattr(self, name, h)

                name = 'h_jetcone_{}Pt_JetPt_R{}_{}'.format(observable + str(ipoint), jetR, trk_thrd)
                pt_bins = linbins(0,200,200)
                ptRL_bins = logbins(1E-3,1E2,60)
                h = ROOT.TH2D(name, name, 200, pt_bins, 60, ptRL_bins)
                h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
                setattr(self, name, h)

            if 'EEC_noweight' in observable or 'EEC_weight2' in observable:
              name = 'h_jetcone_{}_JetPt_R{}_{}'.format(observable, jetR, obs_label)
              pt_bins = linbins(0,200,200)
              RL_bins = logbins(1E-4,1,50)
              h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
              h.GetXaxis().SetTitle('p_{T,ch jet}')
              h.GetYaxis().SetTitle('R_{L}')
              setattr(self, name, h)
                    
  #---------------------------------------------------------------
  # Calculate pair distance of two fastjet particles
  #---------------------------------------------------------------
  def calculate_distance(self, p0, p1):   
    dphiabs = math.fabs(p0.phi() - p1.phi())
    dphi = dphiabs

    if dphiabs > math.pi:
      dphi = 2*math.pi - dphiabs

    deta = p0.eta() - p1.eta()
    return math.sqrt(deta*deta + dphi*dphi)

  def is_same_charge(self, corr_builder, ipoint, constituents, index):
    part1 = int(corr_builder.correlator(ipoint).indices1()[index])
    part2 = int(corr_builder.correlator(ipoint).indices2()[index])
    q1 = int(constituents[part1].python_info().charge)
    q2 = int(constituents[part2].python_info().charge)

    if q1*q2 > 0:
      return True
    else:
      return False
  
  def check_pair_type(self, corr_builder, ipoint, constituents, index):
    part1 = int(corr_builder.correlator(ipoint).indices1()[index])
    part2 = int(corr_builder.correlator(ipoint).indices2()[index])
    type1 = constituents[part1].user_index()
    type2 = constituents[part2].user_index()

    # NB: match the strings in self.pair_type_label = ['bb','sb','ss']
    if type1*type2 >= 0:
      if type1 < 0 or type2 < 0:
        # print('bkg-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
        return 0 # means bkg-bkg
      else:
        # print('sig-sig (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
        return 2 # means sig-sig
    else:
      # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
      return 1 # means sig-bkg

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  #---------------------------------------------------------------
  def fill_jet_histograms(self, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                          obs_label, jet_pt_ungroomed, suffix):

    constituents = fj.sorted_by_pt(jet.constituents())
    c_select = fj.vectorPJ()
    trk_thrd = obs_setting

    for c in constituents:
      if c.pt() < trk_thrd:
        break
      c_select.append(c) # NB: use the break statement since constituents are already sorted
    nconst_jet = len(c_select)

    if self.ENC_pair_cut:
      dphi_cut = -9999 # means no dphi cut
      deta_cut = 0.008
    else:
      dphi_cut = -9999
      deta_cut = -9999

    hname = 'h_{}_JetPt_R{}_{}'
    if self.do_rho_subtraction:
      jet_pt = jet_pt_ungroomed # jet_pt_ungroomed stores subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
    else:
      jet_pt = jet.perp()
    # print('unsubtracted pt',jet.perp(),'subtracted',jet_pt,'# of constituents >',trk_thrd,'is',len(c_select))
    new_corr = ecorrel.CorrelatorBuilder(c_select, jet_pt, 2, 1, dphi_cut, deta_cut)
    for observable in self.observable_list:
      if 'ENC' in observable or 'EEC_noweight' in observable or 'EEC_weight2' in observable:
        for ipoint in range(2, 3):
          for index in range(new_corr.correlator(ipoint).rs().size()):

            # processing only like-sign pairs when self.ENC_pair_like is on
            if self.ENC_pair_like and (not self.is_same_charge(new_corr, ipoint, c_select, index)):
              continue

            # processing only unlike-sign pairs when self.ENC_pair_unlike is on
            if self.ENC_pair_unlike and self.is_same_charge(new_corr, ipoint, c_select, index):
              continue

            if 'ENC' in observable:
              getattr(self, hname.format(observable + str(ipoint), jetR, obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])
              getattr(self, hname.format(observable + str(ipoint) + 'Pt', jetR, obs_label)).Fill(jet_pt, jet_pt*new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index]) # NB: fill pt*RL

            if ipoint==2 and 'EEC_noweight' in observable:
              getattr(self, hname.format(observable, jetR, obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index])

            if ipoint==2 and 'EEC_weight2' in observable:
              getattr(self, hname.format(observable, jetR, obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], pow(new_corr.correlator(ipoint).weights()[index],2))

      if 'jet_pt' in observable:
        getattr(self, hname.format(observable, jetR, obs_label)).Fill(jet_pt) 
        getattr(self, hname.format('Nconst', jetR, obs_label)).Fill(jet_pt, nconst_jet)  
          
  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  #---------------------------------------------------------------
  def fill_perp_cone_histograms(self, cone_parts, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_ungroomed, suffix, rho_bge = 0):

    # calculate perp cone pt after subtraction
    cone_px = 0
    cone_py = 0
    cone_npart = 0
    for part in cone_parts:
      cone_px = cone_px + part.px()
      cone_py = cone_py + part.py()
      cone_npart = cone_npart + 1
    cone_pt = math.sqrt(cone_px*cone_px + cone_py*cone_py)
    # print('cone pt', cone_pt-rho_bge*jet.area(), '(', cone_pt, ')')
    cone_pt = cone_pt-rho_bge*jet.area()
    # print('jet pt', jet_pt_ungroomed, '(', jet.perp(), ')')

    # combine sig jet and perp cone with trk threshold cut
    trk_thrd = obs_setting
    c_combined_select = fj.vectorPJ()

    constituents = fj.sorted_by_pt(jet.constituents())
    # print('jet nconst:',len(constituents))
    for c in constituents:
      if c.pt() < trk_thrd:
        break
      c.set_user_index(1) # positive index for jet constituents
      c_combined_select.append(c) # NB: use the break statement since constituents are already sorted

    nconst_jet = len(c_combined_select)
    # print('jet nconst (with thrd cut):',nconst_jet)

    cone_parts_sorted = fj.sorted_by_pt(cone_parts)
    # print('perp cone nconst:',len(cone_parts_sorted))
    for part in cone_parts_sorted:
      if part.pt() < trk_thrd:
        break
      part.set_user_index(-1) # negative index for perp cone
      c_combined_select.append(part) # NB: use the break statement since constituents are already sorted

    nconst_perp = len(c_combined_select) - nconst_jet
    # print('perp cone nconst (with thrd cut):',nconst_perp)

    if self.ENC_pair_cut:
      dphi_cut = -9999 # means no dphi cut
      deta_cut = 0.008
    else:
      dphi_cut = -9999
      deta_cut = -9999

    hname = 'h_perpcone_{}_JetPt_R{}_{}'
    if self.do_rho_subtraction:
      jet_pt = jet_pt_ungroomed # jet_pt_ungroomed stores subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
    else:
      jet_pt = jet.perp()

    new_corr = ecorrel.CorrelatorBuilder(c_combined_select, jet_pt, 2, 1, dphi_cut, deta_cut)
    for observable in self.observable_list:

      if 'jet_pt' in observable:
        getattr(self, hname.format('pt', jetR, obs_label)).Fill(cone_pt)
        getattr(self, hname.format('Nconst', jetR, obs_label)).Fill(jet_pt, nconst_perp)

      if 'ENC' in observable or 'EEC_noweight' in observable or 'EEC_weight2' in observable:
        for ipoint in range(2, 3):
          for index in range(new_corr.correlator(ipoint).rs().size()):

            # processing only like-sign pairs when self.ENC_pair_like is on
            if self.ENC_pair_like and (not self.is_same_charge(new_corr, ipoint, c_combined_select, index)):
              continue

            # processing only unlike-sign pairs when self.ENC_pair_unlike is on
            if self.ENC_pair_unlike and self.is_same_charge(new_corr, ipoint, c_combined_select, index):
              continue

            # separate out sig-sig, sig-bkg, bkg-bkg correlations for EEC pairs
            pair_type_label = ''
            if self.do_rho_subtraction:
              pair_type = self.check_pair_type(new_corr, ipoint, c_combined_select, index)
              pair_type_label = self.pair_type_labels[pair_type]

            if 'ENC' in observable:
              # print('hname is',hname.format(observable + str(ipoint) + pair_type_label, jetR, obs_label))
              getattr(self, hname.format(observable + str(ipoint) + pair_type_label, jetR, obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])
              getattr(self, hname.format(observable + str(ipoint) + pair_type_label + 'Pt', jetR, obs_label)).Fill(jet_pt, jet_pt*new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index]) # NB: fill pt*RL

            if ipoint==2 and 'EEC_noweight' in observable:
              getattr(self, hname.format(observable + pair_type_label, jetR, obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index])

            if ipoint==2 and 'EEC_weight2' in observable:
              getattr(self, hname.format(observable + pair_type_label, jetR, obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], pow(new_corr.correlator(ipoint).weights()[index],2))

#---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  #---------------------------------------------------------------
  def fill_jet_cone_histograms(self, cone_parts, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_ungroomed, suffix, rho_bge = 0):

    # combine sig jet and perp cone with trk threshold cut
    trk_thrd = obs_setting
    c_select = fj.vectorPJ()

    cone_parts_sorted = fj.sorted_by_pt(cone_parts)
    # print('perp cone nconst:',len(cone_parts_sorted))
    for part in cone_parts_sorted:
      if part.pt() < trk_thrd:
        break
      c_select.append(part) # NB: use the break statement since constituents are already sorted

    if self.ENC_pair_cut:
      dphi_cut = -9999 # means no dphi cut
      deta_cut = 0.008
    else:
      dphi_cut = -9999
      deta_cut = -9999

    hname = 'h_jetcone_{}_JetPt_R{}_{}'
    if self.do_rho_subtraction:
      jet_pt = jet_pt_ungroomed # jet_pt_ungroomed stores subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
    else:
      jet_pt = jet.perp()

    new_corr = ecorrel.CorrelatorBuilder(c_select, jet_pt, 2, 1, dphi_cut, deta_cut)
    for observable in self.observable_list:

      if 'ENC' in observable or 'EEC_noweight' in observable or 'EEC_weight2' in observable:
        for ipoint in range(2, 3):
          for index in range(new_corr.correlator(ipoint).rs().size()):

            # processing only like-sign pairs when self.ENC_pair_like is on
            if self.ENC_pair_like and (not self.is_same_charge(new_corr, ipoint, c_select, index)):
              continue

            # processing only unlike-sign pairs when self.ENC_pair_unlike is on
            if self.ENC_pair_unlike and self.is_same_charge(new_corr, ipoint, c_select, index):
              continue

            if 'ENC' in observable:
              # print('hname is',hname.format(observable + str(ipoint), jetR, obs_label))
              getattr(self, hname.format(observable + str(ipoint), jetR, obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])
              getattr(self, hname.format(observable + str(ipoint) + 'Pt', jetR, obs_label)).Fill(jet_pt, jet_pt*new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index]) # NB: fill pt*RL

            if ipoint==2 and 'EEC_noweight' in observable:
              getattr(self, hname.format(observable, jetR, obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index])

            if ipoint==2 and 'EEC_weight2' in observable:
              getattr(self, hname.format(observable, jetR, obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], pow(new_corr.correlator(ipoint).weights()[index],2))

##################################################################
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Process data')
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
  print('----------------------------------------------------------------')
  
  # If invalid inputFile is given, exit
  if not os.path.exists(args.inputFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.inputFile))
    sys.exit(0)
  
  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  analysis = ProcessData_ENC(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_data()
