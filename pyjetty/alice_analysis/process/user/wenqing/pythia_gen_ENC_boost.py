#!/usr/bin/env python

from __future__ import print_function

import fastjet as fj
import fjcontrib
import fjext

import ROOT

import tqdm
import yaml
import copy
import argparse
import os
import array
import numpy as np
import math

from pyjetty.mputils import *

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext
import pythiaext
import ecorrel

from pyjetty.alice_analysis.process.base import process_base
from pyjetty.mputils.csubtractor import CEventSubtractor

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)
# Automatically set Sumw2 when creating new histograms
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

################################################################
class PythiaGenENCBoost(process_base.ProcessBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

        super(PythiaGenENCBoost, self).__init__(
            input_file, config_file, output_dir, debug_level, **kwargs)

        # Call base class initialization
        process_base.ProcessBase.initialize_config(self)

        # Read config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.jetR_list = config["jetR"] 

        self.nev = args.nev

        # particle level - ALICE tracking restriction
        self.max_eta_hadron = 0.9

        if 'rm_trk_min_pt' in config:
            self.rm_trk_min_pt = config['rm_trk_min_pt']
        else:
            self.rm_trk_min_pt = False

        if 'jet_matching_distance' in config:
            self.jet_matching_distance = config['jet_matching_distance']
        else:
            self.jet_matching_distance = 0.6 # default to 0.6

        if 'mc_fraction_threshold' in config:
            self.mc_fraction_threshold = config['mc_fraction_threshold']
        else:
            self.mc_fraction_threshold = 0.5 # default to 0.5

        # perp cone settings
        if 'static_perpcone' in config:
            self.static_perpcone = config['static_perpcone']
        else:
            self.static_perpcone = True # NB: set default to rigid cone (less fluctuations)

        # perp and jet cone sizes
        self.coneR_list = config["coneR"] 

        # ENC settings
        if 'thrd' in config:
            self.thrd_list = config['thrd']
        else:
            self.thrd_list = [1.0]
        
        self.dphi_cut = -9999
        self.deta_cut = -9999
        self.npoint = 2
        self.npower = 1

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def pythia_parton_hadron(self, args):
 
        # Create ROOT TTree file for storing raw PYTHIA particle information
        outf_path = os.path.join(self.output_dir, args.tree_output_fname)
        outf = ROOT.TFile(outf_path, 'recreate')
        outf.cd()

        mycfg = []
        pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)

        # Initialize response histograms
        self.initialize_hist()

        # print the banner first
        fj.ClusterSequence.print_banner()
        print()

        self.init_jet_tools()
        self.analyze_events(pythia)
        pythia.stat()
        print()
        
        self.scale_print_final_info(pythia)

        outf.Write()
        outf.Close()

        self.save_output_objects()

    #---------------------------------------------------------------
    # Initialize histograms
    #---------------------------------------------------------------
    def initialize_hist(self):

        self.hNevents = ROOT.TH1I("hNevents", 'Number accepted events (unscaled)', 2, -0.5, 1.5)

        for jetR in self.jetR_list:

            # Store a list of all the histograms just so that we can rescale them later
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
            setattr(self, hist_list_name, [])

            R_label = str(jetR).replace('.', '') + 'Scaled'

            name = 'h_JetPt_ch_pp_R{}'.format(R_label)
            pt_bins = linbins(0,1000,500)
            h = ROOT.TH1D(name, name, 500, pt_bins)
            h.GetYaxis().SetTitle('p_{T, pp jet}')
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)

            # ENC histograms
            for ipoint in range(2, self.npoint+1):

                pt_bins = linbins(0,200,200)
                RL_bins = logbins(1E-4,1,50)

                name = 'h_ENC{}_JetPt_ch_R{}'.format(str(ipoint), R_label)
                print('Initialize histogram',name)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                h.GetXaxis().SetTitle('p_{T, pp jet}')
                h.GetYaxis().SetTitle('R_{L}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

    #---------------------------------------------------------------
    # Initiate jet defs, selectors, and sd (if required)
    #---------------------------------------------------------------
    def init_jet_tools(self):
        
        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')      
            
            # set up our jet definition and a jet selector
            # NB: area calculation enabled
            jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
            setattr(self, "jet_def_R%s" % jetR_str, jet_def)
            print(jet_def)

        # pwarning('max eta for particles after hadronization set to', self.max_eta_hadron)
        if self.rm_trk_min_pt:
            track_selector_ch = fj.SelectorPtMin(0)
        else:
            track_selector_ch = fj.SelectorPtMin(0.15)

        setattr(self, "track_selector_ch", track_selector_ch)

        pfc_selector1 = fj.SelectorPtMin(1.)
        setattr(self, "pfc_def_10", pfc_selector1)

        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')
            
            jet_selector = fj.SelectorPtMin(5) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)
            setattr(self, "jet_selector_R%s" % jetR_str, jet_selector)

    #---------------------------------------------------------------
    # Analyze events and pass information on to jet finding
    #---------------------------------------------------------------
    def analyze_events(self, pythia):
        
        iev = 0  # Event loop count

        while iev < self.nev:
            if iev % 100 == 0:
                print('ievt',iev)

            if not pythia.next():
                continue

            self.event = pythia.event
            # print(self.event)

            # charged particle level
            self.parts_pythia_ch = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)

            # Some "accepted" events don't survive hadronization step -- keep track here
            self.hNevents.Fill(0)

            self.analyze_jets()

            iev += 1

    def calculate_boost_vec(self, jet_before, new_pt):
        
        print('before boost',jet_before.pt(), jet_before.rapidity(), jet_before.phi(), jet_before.m())
        rest_vec_before = jet_before._jet_rest
        jet_after = fj.PseudoJet()
        jet_after.reset_PtYPhiM(new_pt, jet_before.rapidity(), jet_before.phi(), jet_before.m())
        rest_vec_after = jet_after._jet_rest

        jet_after_check = fj.PseudoJet()
        jet_after_check.reset_PtYPhiM(jet_before.pt(), jet_before.rapidity(), jet_before.phi(), jet_before.m())
        jet_after_check.Boost(-rest_vec_before+rest_vec_after)
        print('after boost',jet_after_check.pt(), jet_after_check.rapidity(), jet_after_check.phi(), jet_after_check.m())

        return -rest_vec_before+rest_vec_after

    #---------------------------------------------------------------
    # Create a copy of list of particles
    #---------------------------------------------------------------
    def copy_parts(self, parts, remove_ghosts = True):
    # don't need to re-init every part for a deep copy
    # the last arguement enable/disable the removal of ghost particles from jet area calculation (default set to true)
        parts_copied = fj.vectorPJ()
        for part in parts:
          if remove_ghosts:
            if part.pt() > 0.01:
              parts_copied.push_back(part)
          else:
            parts_copied.push_back(part)

        return parts_copied

    #---------------------------------------------------------------
    # Lorentz boost the list of particles
    #---------------------------------------------------------------
    def boost_parts(parts_before, boost_vec):
        parts_after = fj.vectorPJ()
        for part in parts_before:
            # print('before',part.pt(),part.phi(),part.rapidity())
            part.Boost(boost_vec)
            # print('after',part.pt(),part.phi(),part.rapidity())
            parts_after.push_back(part)

        return parts_after

    #---------------------------------------------------------------
    # Find jets, do matching between levels, and fill histograms & trees
    #---------------------------------------------------------------
    def analyze_jets(self):
        # Loop over jet radii
        for jetR in self.jetR_list:

            jetR_str = str(jetR).replace('.', '')
            jet_selector = getattr(self, "jet_selector_R%s" % jetR_str)
            jet_def = getattr(self, "jet_def_R%s" % jetR_str)
            track_selector_ch = getattr(self, "track_selector_ch")

            cs_pp = fj.ClusterSequence(track_selector_ch(self.parts_pythia_ch), jet_def)
            jets_pp = fj.sorted_by_pt( jet_selector(cs_pp.inclusive_jets()) )

            #-------------------------------------------------------------
            # loop over jets and fill EEC histograms with jet constituents
            for jet_pp in jets_pp:
                if jet_pp.perp() > 20 and jet_pp.perp() < 21:
                    ref_pt = 40
                    self.boost_vec = calculate_boost_vec(jet_pp, ref_pt)
                    parts_in_jet = self.copy_parts(constituents)
                    self.boost_parts(parts_in_jet, boost_vec)

                # hname = 'h_JetPt_ch_pp_R{}'.format(R_label)
                # getattr(self, hname).Fill(jet_pp.perp())
                # hname = 'h_ENC{{}}_JetPt_ch_R{}_{{}}'.format(R_label)
                # self.fill_jet_histograms(hname, jet_pp)

    # #---------------------------------------------------------------
    # # Fill jet constituents for unmatched jets
    # #---------------------------------------------------------------
    # def fill_jet_histograms(self, hname, jet):

    #     constituents = fj.sorted_by_pt(jet.constituents())

    #     for thrd in self.thrd_list:
    #         c_select = fj.vectorPJ()
    #         thrd_label = 'trk{:.0f}'.format(thrd*10)
    #         for c in constituents:
    #           if c.pt() < thrd:
    #             break
    #           c_select.append(c) # NB: use the break statement since constituents are already sorted

    #         dphi_cut = -9999
    #         deta_cut = -9999
    #         new_corr = ecorrel.CorrelatorBuilder(c_select, jet.perp(), self.npoint, self.npower, dphi_cut, deta_cut)

    #         for ipoint in range(2, self.npoint+1):
    #             for index in range(new_corr.correlator(ipoint).rs().size()):              
    #                 getattr(self,hname.format(ipoint, thrd_label)).Fill(jet.perp(), new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])

    #---------------------------------------------------------------
    # Initiate scaling of all histograms and print final simulation info
    #---------------------------------------------------------------
    def scale_print_final_info(self, pythia):
        # Scale all jet histograms by the appropriate factor from generated cross section and the number of accepted events
        scale_f = pythia.info.sigmaGen() / self.hNevents.GetBinContent(1)
        print("scaling factor is",scale_f)

        for jetR in self.jetR_list:
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '') 
            for h in getattr(self, hist_list_name):
                h.Scale(scale_f)

        print("N total final events:", int(self.hNevents.GetBinContent(1)), "with",
              int(pythia.info.nAccepted() - self.hNevents.GetBinContent(1)),
              "events rejected at hadronization step")
        self.hNevents.SetBinError(1, 0)

################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly',
                                     prog=os.path.basename(__file__))
    pyconf.add_standard_pythia_args(parser)
    # Could use --py-seed
    parser.add_argument('-o', '--output-dir', action='store', type=str, default='./', 
                        help='Output directory for generated ROOT file(s)')
    parser.add_argument('--tree-output-fname', default="AnalysisResults.root", type=str,
                        help="Filename for the (unscaled) generated particle ROOT TTree")
    parser.add_argument('-c', '--config_file', action='store', type=str, default='config/analysis_config.yaml',
                        help="Path of config file for observable configurations")

    args = parser.parse_args()

    # If invalid configFile is given, exit
    if not os.path.exists(args.config_file):
        print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    # Have at least 1 event
    if args.nev < 1:
        args.nev = 1

    process = PythiaGenENCBoost(config_file=args.config_file, output_dir=args.output_dir, args=args)
    process.pythia_parton_hadron(args)