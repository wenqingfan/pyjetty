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
class PythiaGenENC(process_base.ProcessBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

        super(PythiaGenENC, self).__init__(
            input_file, config_file, output_dir, debug_level, **kwargs)

        # Call base class initialization
        process_base.ProcessBase.initialize_config(self)

        # Read config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.jet_levels = config["jet_levels"] # levels = ["p", "h", "ch"]

        self.jetR_list = config["jetR"] 

        self.nev = args.nev

        # particle level - ALICE tracking restriction
        self.max_eta_hadron = 0.9

        if 'beyond_jetR' in config:
            self.beyond_jetR = config['beyond_jetR']
        else:
            self.beyond_jetR = False

        self.ref_jet_level = "ch"
        self.ref_jetR = 0.4 # hard coded for now 
        self.part_levels = config["part_levels"] 

        # ENC settings
        self.dphi_cut = -9999
        self.deta_cut = -9999
        self.npoint = 2
        self.npower = 1

        if 'do_matching' in config:
            self.do_matching = config['do_matching']
        else:
            self.do_matching = False

        if 'matched_jet_type' in config:
            self.matched_jet_type = config['matched_jet_type']
        else:
            self.matched_jet_type = 'ch' # default matched jet type set to charged jets

        # if matched jet type is parton and the following variable is True, analyze the matched histograms with leading parton information
        if 'use_leading_parton' in config:
            self.use_leading_parton = config['use_leading_parton']
        else:
            self.use_leading_parton = False

        # whether to use matched reference jet for jet selection (default to True)
        if 'use_ref_for_jet_selection' in config:
            self.use_ref_for_jet_selection = config['use_ref_for_jet_selection']
        else:
            self.use_ref_for_jet_selection = True

        # whether to use matched reference jet for pt scaling (default to True)
        if 'use_ref_for_pt_scaling' in config:
            self.use_ref_for_pt_scaling = config['use_ref_for_pt_scaling']
        else:
            self.use_ref_for_pt_scaling = True

        self.jet_matching_distance = config["jet_matching_distance"] 

        if 'do_gluon_jet' in config:
            self.do_gluon_jet = config['do_gluon_jet']
        else:
            self.do_gluon_jet = False

        if 'do_quark_jet' in config:
            self.do_quark_jet = config['do_quark_jet']
        else:
            self.do_quark_jet = False

        if 'do_tagging' in config:
            self.do_tagging = config['do_tagging']
        else:
            self.do_tagging = False

        if 'do_reshuffle' in config:
            self.do_reshuffle = config['do_reshuffle']
        else:
            self.do_reshuffle = False

        if 'do_theory_check' in config:
            self.do_theory_check = config['do_theory_check']
        else:
            self.do_theory_check = False

        if 'rm_trk_min_pt' in config:
            self.rm_trk_min_pt = config['rm_trk_min_pt']
        else:
            self.rm_trk_min_pt = False

        if 'leading_pt' in config:
            self.leading_pt = config['leading_pt']
        else:
            self.leading_pt = -1 # negative means no leading track cut

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def pythia_parton_hadron(self, args):
 
        # Create ROOT TTree file for storing raw PYTHIA particle information
        outf_path = os.path.join(self.output_dir, args.tree_output_fname)
        outf = ROOT.TFile(outf_path, 'recreate')
        outf.cd()

        mycfg = []
        mycfg.append("HadronLevel:all=off")
        pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)

        # Initialize response histograms
        self.initialize_hist()

        # print the banner first
        fj.ClusterSequence.print_banner()
        print()

        self.init_jet_tools()
        self.calculate_events(pythia)
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

            for jet_level in self.jet_levels:
                # ENC histograms (jet level == part level)
                for ipoint in range(2, self.npoint+1):
                    name = 'h_ENC{}_JetPt_{}_R{}_trk00'.format(str(ipoint), jet_level, R_label)
                    print('Initialize histogram',name)
                    pt_bins = linbins(0,200,200)
                    RL_bins = logbins(1E-4,1,50)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('pT (jet)')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    name = 'h_ENC{}_JetPt_{}_R{}_trk10'.format(str(ipoint), jet_level, R_label)
                    print('Initialize histogram',name)
                    pt_bins = linbins(0,200,200)
                    RL_bins = logbins(1E-4,1,50)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('pT (jet)')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    name = 'h_ENC{}Pt_JetPt_{}_R{}_trk00'.format(str(ipoint), jet_level, R_label)
                    print('Initialize histogram',name)
                    pt_bins = linbins(0,200,200)
                    ptRL_bins = logbins(1E-3,1E2,60)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 60, ptRL_bins)
                    h.GetXaxis().SetTitle('p_{T,ch jet}')
                    h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    name = 'h_ENC{}Pt_JetPt_{}_R{}_trk10'.format(str(ipoint), jet_level, R_label)
                    print('Initialize histogram',name)
                    ptRL_bins = logbins(1E-3,1E2,60)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 60, ptRL_bins)
                    h.GetXaxis().SetTitle('p_{T,ch jet}')
                    h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    if self.do_reshuffle:
                        name = 'h_reshuffle_ENC{}_JetPt_{}_R{}_trk00'.format(str(ipoint), jet_level, R_label)
                        print('Initialize histogram',name)
                        pt_bins = linbins(0,200,200)
                        RL_bins = logbins(1E-4,1,50)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('pT (jet)')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        name = 'h_reshuffle_ENC{}_JetPt_{}_R{}_trk10'.format(str(ipoint), jet_level, R_label)
                        print('Initialize histogram',name)
                        pt_bins = linbins(0,200,200)
                        RL_bins = logbins(1E-4,1,50)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('pT (jet)')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                    # only save charge separation for pT>1GeV for now
                    if jet_level == "ch":
                        name = 'h_ENC{}_JetPt_{}_R{}_unlike_trk10'.format(str(ipoint), jet_level, R_label)
                        print('Initialize histogram',name)
                        pt_bins = linbins(0,200,200)
                        RL_bins = logbins(1E-4,1,50)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('pT (jet)')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        name = 'h_ENC{}_JetPt_{}_R{}_like_trk10'.format(str(ipoint), jet_level, R_label)
                        print('Initialize histogram',name)
                        pt_bins = linbins(0,200,200)
                        RL_bins = logbins(1E-4,1,50)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('pT (jet)')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                # Jet pt vs N constituents
                name = 'h_Nconst_JetPt_{}_R{}_trk00'.format(jet_level, R_label)
                print('Initialize histogram',name)
                pt_bins = linbins(0,200,200)
                Nconst_bins = linbins(0,50,50)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, Nconst_bins)
                h.GetXaxis().SetTitle('pT (jet)')
                h.GetYaxis().SetTitle('N_{const}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                name = 'h_Nconst_JetPt_{}_R{}_trk10'.format(jet_level, R_label)
                print('Initialize histogram',name)
                pt_bins = linbins(0,200,200)
                Nconst_bins = linbins(0,50,50)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, Nconst_bins)
                h.GetXaxis().SetTitle('pT (jet)')
                h.GetYaxis().SetTitle('N_{const}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                # NB: Only do the cone check for one reference radius and charged jets for now
                if self.beyond_jetR and (jetR == self.ref_jetR) and (jet_level == self.ref_jet_level):
                    for part_level in self.part_levels:
                        for ipoint in range(2, self.npoint+1):
                            name = 'h_ENC{}_cone_max_JetPt_{}_R{}_{}_trk00'.format(str(ipoint), jet_level, R_label, part_level)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            RL_bins = logbins(1E-4,1,50)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('pT (jet)')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_ENC{}_cone_max_JetPt_{}_R{}_{}_trk10'.format(str(ipoint), jet_level, R_label, part_level)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            RL_bins = logbins(1E-4,1,50)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('pT (jet)')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_ENC{}_cone_jetR_JetPt_{}_R{}_{}_trk00'.format(str(ipoint), jet_level, R_label, part_level)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            RL_bins = logbins(1E-4,1,50)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('pT (jet)')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_ENC{}_cone_jetR_JetPt_{}_R{}_{}_trk10'.format(str(ipoint), jet_level, R_label, part_level)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            RL_bins = logbins(1E-4,1,50)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('pT (jet)')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

            if self.do_matching and (jetR == self.ref_jetR):
                name = 'h_matched_JetPt_ch_vs_p_R{}'.format(R_label)
                pt_bins = linbins(0,200,200)
                h = ROOT.TH2D(name, name, 200, pt_bins, 200, pt_bins)
                h.GetXaxis().SetTitle('p_{T,ch jet}')
                h.GetYaxis().SetTitle('p_{T,p}')
                setattr(self, name, h)

                name = 'h_matched_JetPt_h_vs_p_R{}'.format(R_label)
                pt_bins = linbins(0,200,200)
                h = ROOT.TH2D(name, name, 200, pt_bins, 200, pt_bins)
                h.GetXaxis().SetTitle('p_{T,h jet}')
                h.GetYaxis().SetTitle('p_{T,p}')
                setattr(self, name, h)

                name = 'h_matched_JetPt_ch_vs_h_R{}'.format(R_label)
                pt_bins = linbins(0,200,200)
                h = ROOT.TH2D(name, name, 200, pt_bins, 200, pt_bins)
                h.GetXaxis().SetTitle('p_{T,ch jet}')
                h.GetYaxis().SetTitle('p_{T,h jet}')
                setattr(self, name, h)

                name = 'h_matched_JetPt_p_over_ch_ratio_R{}'.format(R_label)
                ratio_bins = linbins(0,10,500)
                pt_bins = linbins(0,200,200)
                h = ROOT.TH2D(name, name, 200, ratio_bins, 200, pt_bins)
                h.GetXaxis().SetTitle('p_{T,p}/p_{T,ch jet}') # this ratio should be mostly within [0,1]
                h.GetYaxis().SetTitle('p_{T,ch jet}')
                setattr(self, name, h)

                name = 'h_matched_JetPt_p_over_h_ratio_R{}'.format(R_label)
                ratio_bins = linbins(0,10,500)
                pt_bins = linbins(0,200,200)
                h = ROOT.TH2D(name, name, 200, ratio_bins, 200, pt_bins)
                h.GetXaxis().SetTitle('p_{T,p}/p_{T,h jet}')  # this ratio should be mostly within [0,1]
                h.GetYaxis().SetTitle('p_{T,h jet}')
                setattr(self, name, h)

                if self.use_leading_parton:
                    name = 'h_matched_JetPt_p_vs_p_R{}'.format(R_label)
                    pt_bins = linbins(0,200,200)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 200, pt_bins)
                    h.GetXaxis().SetTitle('p_{T,p jet}')
                    h.GetYaxis().SetTitle('p_{T,p}')
                    setattr(self, name, h)

                    name = 'h_matched_JetPt_p_over_p_ratio_R{}'.format(R_label)
                    ratio_bins = linbins(0,10,500)
                    pt_bins = linbins(0,200,200)
                    h = ROOT.TH2D(name, name, 200, ratio_bins, 200, pt_bins)
                    h.GetXaxis().SetTitle('p_{T,p}/p_{T,p jet}')  # this ratio should be mostly within [0,1]
                    h.GetYaxis().SetTitle('p_{T,p jet}')
                    setattr(self, name, h)

                for jet_level in ['p', 'h', 'ch']:
                    tag_levels = ['']
                    if self.do_tagging:
                        tag_levels = ['-1', '1', '2', '3', '4', '5', '6', '21']
                    for tag_level in tag_levels:
                        for ipoint in range(2, self.npoint+1):
                            name = 'h_matched_ENC{}_JetPt_{}{}_R{}_trk00'.format(str(ipoint), jet_level, tag_level, R_label)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            RL_bins = logbins(1E-4,1,50)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('pT (jet)')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_matched_ENC{}_JetPt_{}{}_R{}_trk10'.format(str(ipoint), jet_level, tag_level, R_label)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            RL_bins = logbins(1E-4,1,50)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('pT (jet)')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_matched_ENC{}Pt_JetPt_{}{}_R{}_trk00'.format(str(ipoint), jet_level, tag_level, R_label)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            ptRL_bins = logbins(1E-3,1E2,60)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 60, ptRL_bins)
                            h.GetXaxis().SetTitle('p_{T,ch jet}')
                            h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_matched_ENC{}Pt_JetPt_{}{}_R{}_trk10'.format(str(ipoint), jet_level, tag_level, R_label)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            ptRL_bins = logbins(1E-3,1E2,60)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 60, ptRL_bins)
                            h.GetXaxis().SetTitle('p_{T,ch jet}')
                            h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                        # Jet pt vs N constituents
                        name = 'h_matched_Nconst_JetPt_{}{}_R{}_trk00'.format(jet_level, tag_level, R_label)
                        print('Initialize histogram',name)
                        pt_bins = linbins(0,200,200)
                        Nconst_bins = linbins(0,50,50)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, Nconst_bins)
                        h.GetXaxis().SetTitle('pT (jet)')
                        h.GetYaxis().SetTitle('N_{const}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        name = 'h_matched_Nconst_JetPt_{}{}_R{}_trk10'.format(jet_level, tag_level, R_label)
                        print('Initialize histogram',name)
                        pt_bins = linbins(0,200,200)
                        Nconst_bins = linbins(0,50,50)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, Nconst_bins)
                        h.GetXaxis().SetTitle('pT (jet)')
                        h.GetYaxis().SetTitle('N_{const}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

    #---------------------------------------------------------------
    # Initiate jet defs, selectors, and sd (if required)
    #---------------------------------------------------------------
    def init_jet_tools(self):
        
        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')      
            
            # set up our jet definition and a jet selector
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
            
            jet_selector = fj.SelectorPtMin(5) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR) # FIX ME: use 5 or lower? use it on all ch, h, p jets?
            setattr(self, "jet_selector_R%s" % jetR_str, jet_selector)

    #---------------------------------------------------------------
    # Calculate events and pass information on to jet finding
    #---------------------------------------------------------------
    def calculate_events(self, pythia):
        
        iev = 0  # Event loop count

        while iev < self.nev:
            if iev % 100 == 0:
                print('ievt',iev)

            if not pythia.next():
                continue

            self.event = pythia.event
            # print(self.event)

            leading_parton1 = fj.PseudoJet(pythia.event[5].px(),pythia.event[5].py(),pythia.event[5].pz(),pythia.event[5].e())
            leading_parton2 = fj.PseudoJet(pythia.event[6].px(),pythia.event[6].py(),pythia.event[6].pz(),pythia.event[6].e())

            # save index of the leading parton inside this event
            leading_parton1.set_user_index(5)
            leading_parton2.set_user_index(6)

            # print('---------------------------------')
            # print('parton 1',leading_parton1.user_index(),'pt',leading_parton1.perp(),'phi',leading_parton1.phi(),'eta',leading_parton1.eta())
            # print('parton 2',leading_parton2.user_index(),'pt',leading_parton2.perp(),'phi',leading_parton2.phi(),'eta',leading_parton2.eta())
            
            self.parton_parents = [leading_parton1, leading_parton2]

            self.parts_pythia_p = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True) # final stable partons

            hstatus = pythia.forceHadronLevel()
            if not hstatus:
                continue

            # full particle level
            self.parts_pythia_h = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True)

            # charged particle level
            self.parts_pythia_ch = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)

            # Some "accepted" events don't survive hadronization step -- keep track here
            self.hNevents.Fill(0)

            self.find_jets_fill_trees()

            iev += 1

    def find_parts_around_jets(self, jet, parts_pythia, cone_R):
        # select particles around jet axis
        parts = fj.vectorPJ()
        for part in parts_pythia:
            if jet.delta_R(part) <= cone_R:
                parts.push_back(part)
        
        return parts

    #---------------------------------------------------------------
    # Form EEC using a cone around certain type of jets
    #---------------------------------------------------------------
    def fill_beyond_jet_histograms(self, jet_level, part_level, jet, jetR, R_label):
        # fill EEC histograms for cone around jet axis

        # Get the particles at certain level
        if part_level == "p":
            parts_pythia  = self.parts_pythia_p
        if part_level == "h":
            parts_pythia  = self.parts_pythia_h
        if part_level == "ch":
            parts_pythia  = self.parts_pythia_ch

        pfc_selector1 = getattr(self, "pfc_def_10")

        # select beyond constituents
        _p_select_cone_max = self.find_parts_around_jets(jet, parts_pythia, 1.0) # select within dR < 1
        _p_select_cone_jetR = self.find_parts_around_jets(jet, _p_select_cone_max, jetR) # select within previously selected parts

        _p_select0_cone_max = fj.vectorPJ()
        _ = [_p_select0_cone_max.push_back(p) for p in _p_select_cone_max]

        _p_select1_cone_max = fj.vectorPJ()
        _ = [_p_select1_cone_max.push_back(p) for p in pfc_selector1(_p_select_cone_max)]

        _p_select0_cone_jetR = fj.vectorPJ()
        _ = [_p_select0_cone_jetR.push_back(p) for p in _p_select_cone_jetR]

        _p_select1_cone_jetR = fj.vectorPJ()
        _ = [_p_select1_cone_jetR.push_back(p) for p in pfc_selector1(_p_select_cone_jetR)]

        cb0_cone_max = ecorrel.CorrelatorBuilder(_p_select0_cone_max, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)
        cb1_cone_max = ecorrel.CorrelatorBuilder(_p_select1_cone_max, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        cb0_cone_jetR = ecorrel.CorrelatorBuilder(_p_select0_cone_jetR, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)
        cb1_cone_jetR = ecorrel.CorrelatorBuilder(_p_select1_cone_jetR, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        for ipoint in range(2, self.npoint+1):
            for index in range(cb0_cone_max.correlator(ipoint).rs().size()):
                    getattr(self, 'h_ENC{}_cone_max_JetPt_{}_R{}_{}_trk00'.format(str(ipoint), jet_level, R_label, part_level)).Fill(jet.perp(), cb0_cone_max.correlator(ipoint).rs()[index], cb0_cone_max.correlator(ipoint).weights()[index])
            for index in range(cb1_cone_max.correlator(ipoint).rs().size()):
                    getattr(self, 'h_ENC{}_cone_max_JetPt_{}_R{}_{}_trk10'.format(str(ipoint), jet_level, R_label, part_level)).Fill(jet.perp(), cb1_cone_max.correlator(ipoint).rs()[index], cb1_cone_max.correlator(ipoint).weights()[index])
            for index in range(cb0_cone_jetR.correlator(ipoint).rs().size()):
                    getattr(self, 'h_ENC{}_cone_jetR_JetPt_{}_R{}_{}_trk00'.format(str(ipoint), jet_level, R_label, part_level)).Fill(jet.perp(), cb0_cone_jetR.correlator(ipoint).rs()[index], cb0_cone_jetR.correlator(ipoint).weights()[index])
            for index in range(cb1_cone_jetR.correlator(ipoint).rs().size()):
                    getattr(self, 'h_ENC{}_cone_jetR_JetPt_{}_R{}_{}_trk10'.format(str(ipoint), jet_level, R_label, part_level)).Fill(jet.perp(), cb1_cone_jetR.correlator(ipoint).rs()[index], cb1_cone_jetR.correlator(ipoint).weights()[index])

    def reshuffle_parts(self, parts, jet, jetR):
        # rotate parts in azimuthal direction
        parts_reshuffled = fj.vectorPJ()
        for part in parts:
          pt_new = part.pt()
          m_new = part.m()
          R_new = 1 # initialize to a big radius
          while R_new > jetR:
            phi_new = np.random.uniform(-jetR,+jetR)
            y_new = np.random.uniform(-jetR,+jetR)
            R_new = math.sqrt(phi_new*phi_new+y_new*y_new)
          phi_new = part.phi() + phi_new
          y_new = part.rapidity() + y_new
          
          # print('before',part.phi())
          part.reset_PtYPhiM(pt_new, y_new, phi_new, m_new)
          # print('after',part.phi())
          parts_reshuffled.push_back(part)
        
        return parts_reshuffled

    #---------------------------------------------------------------
    # Form EEC using jet constituents
    #---------------------------------------------------------------
    def fill_jet_histograms(self, level, jet, jetR, R_label):
        # leading track selection
        if self.leading_pt > 0:
            constituents = fj.sorted_by_pt(jet.constituents())
            if constituents[0].perp() < self.leading_pt:
                return

        # fill EEC histograms for jet constituents
        pfc_selector1 = getattr(self, "pfc_def_10")

        # select all constituents with no cut
        _c_select0 = fj.vectorPJ()
        for c in jet.constituents():
            if self.do_theory_check:
                if pythiafjext.getPythia8Particle(c).charge()!=0:
                    _c_select0.push_back(c)
            else:
                _c_select0.push_back(c)
        cb0 = ecorrel.CorrelatorBuilder(_c_select0, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        # select constituents with 1 GeV cut
        _c_select1 = fj.vectorPJ()
        for c in pfc_selector1(jet.constituents()):
            if self.do_theory_check:
                if pythiafjext.getPythia8Particle(c).charge()!=0:
                    _c_select1.push_back(c)
            else:
                _c_select1.push_back(c)
        cb1 = ecorrel.CorrelatorBuilder(_c_select1, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        for ipoint in range(2, self.npoint+1):
            for index in range(cb0.correlator(ipoint).rs().size()):
                    getattr(self, 'h_ENC{}_JetPt_{}_R{}_trk00'.format(str(ipoint), level, R_label)).Fill(jet.perp(), cb0.correlator(ipoint).rs()[index], cb0.correlator(ipoint).weights()[index])
                    getattr(self, 'h_ENC{}Pt_JetPt_{}_R{}_trk00'.format(str(ipoint), level, R_label)).Fill(jet.perp(), jet.perp()*cb0.correlator(ipoint).rs()[index], cb0.correlator(ipoint).weights()[index])
            for index in range(cb1.correlator(ipoint).rs().size()):
                    getattr(self, 'h_ENC{}_JetPt_{}_R{}_trk10'.format(str(ipoint), level, R_label)).Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
                    getattr(self, 'h_ENC{}Pt_JetPt_{}_R{}_trk10'.format(str(ipoint), level, R_label)).Fill(jet.perp(), jet.perp()*cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
            
        if level == "ch":
            for ipoint in range(2, self.npoint+1):
                # only fill trk pt > 1 GeV here for now
                for index in range(cb1.correlator(ipoint).rs().size()):
                    part1 = int(cb1.correlator(ipoint).indices1()[index])
                    part2 = int(cb1.correlator(ipoint).indices2()[index])
                    c1 = _c_select1[part1]
                    c2 = _c_select1[part2]
                    if pythiafjext.getPythia8Particle(c1).charge()*pythiafjext.getPythia8Particle(c2).charge() < 0:
                        # print("unlike-sign pair ",pythiafjext.getPythia8Particle(c1).id(),pythiafjext.getPythia8Particle(c2).id())
                        getattr(self, 'h_ENC{}_JetPt_{}_R{}_unlike_trk10'.format(str(ipoint), level, R_label)).Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
                    else:
                        # print("likesign pair ",pythiafjext.getPythia8Particle(c1).id(),pythiafjext.getPythia8Particle(c2).id())
                        getattr(self, 'h_ENC{}_JetPt_{}_R{}_like_trk10'.format(str(ipoint), level, R_label)).Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])

        getattr(self, 'h_Nconst_JetPt_{}_R{}_trk00'.format(level, R_label)).Fill(jet.perp(), len(_c_select0))
        getattr(self, 'h_Nconst_JetPt_{}_R{}_trk10'.format(level, R_label)).Fill(jet.perp(), len(_c_select1))

        # reshuffled consitituents
        if self.do_reshuffle:
            _c_reshuffle0 = self.reshuffle_parts(_c_select0, jet, jetR)
            cb_reshuffle0 = ecorrel.CorrelatorBuilder(_c_reshuffle0, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)
            _c_reshuffle1 = self.reshuffle_parts(_c_select1, jet, jetR)
            cb_reshuffle1 = ecorrel.CorrelatorBuilder(_c_reshuffle1, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

            for ipoint in range(2, self.npoint+1):
                for index in range(cb_reshuffle0.correlator(ipoint).rs().size()):
                        getattr(self, 'h_reshuffle_ENC{}_JetPt_{}_R{}_trk00'.format(str(ipoint), level, R_label)).Fill(jet.perp(), cb_reshuffle0.correlator(ipoint).rs()[index], cb_reshuffle0.correlator(ipoint).weights()[index])
                for index in range(cb_reshuffle1.correlator(ipoint).rs().size()):
                        getattr(self, 'h_reshuffle_ENC{}_JetPt_{}_R{}_trk10'.format(str(ipoint), level, R_label)).Fill(jet.perp(), cb_reshuffle1.correlator(ipoint).rs()[index], cb_reshuffle1.correlator(ipoint).weights()[index])

    #---------------------------------------------------------------
    # Form EEC using jet constituents for matched jets
    #---------------------------------------------------------------
    def fill_matched_jet_histograms(self, level, jet, ref_jet, R_label):
        # use the jet pt for energy weight but ref_jet pt can be used when fill jet samples into jet pt bins

        # leading track selection
        if self.leading_pt > 0:
            constituents = fj.sorted_by_pt(jet.constituents())
            if constituents[0].perp() < self.leading_pt:
                return
        
        pfc_selector1 = getattr(self, "pfc_def_10")
        # print(level,'with number of constituents',len(jet.constituents()),'(',len(pfc_selector1(jet.constituents())),')')
        # print('jet pt',jet.perp(),'ref jet pt',ref_jet.perp())

        # select all constituents with no cut
        _c_select0 = fj.vectorPJ()
        for c in jet.constituents():
            if self.do_theory_check:
                if pythiafjext.getPythia8Particle(c).charge()!=0:
                    _c_select0.push_back(c)
            else:
                _c_select0.push_back(c)
        cb0 = ecorrel.CorrelatorBuilder(_c_select0, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        # select constituents with 1 GeV cut
        _c_select1 = fj.vectorPJ()
        for c in pfc_selector1(jet.constituents()):
            if self.do_theory_check:
                if pythiafjext.getPythia8Particle(c).charge()!=0:
                    _c_select1.push_back(c)
            else:
                _c_select1.push_back(c)
        cb1 = ecorrel.CorrelatorBuilder(_c_select1, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        if self.do_tagging:
            if jet.user_index()>0:
                leading_parton_id = abs(self.event[jet.user_index()].id())
                if (leading_parton_id>0 and leading_parton_id<7): # quarks (1-6)
                    level=level+str(leading_parton_id)
                    # print('quark', level, 'jet pt', jet.perp(), 'phi', jet.phi(), 'eta', jet.eta(), 'id', leading_parton_id)
                if leading_parton_id==9 or leading_parton_id==21: # gluons
                    level=level+'21'
                    # print('gluon', level, 'jet pt', jet.perp(), 'phi', jet.phi(), 'eta', jet.eta(), 'id', jet.user_index())
            else:
                level=level+'-1'
                # print('Untagged', level, 'jet pt', jet.perp(), 'phi', jet.phi(), 'eta', jet.eta(), 'id', jet.user_index())

        # by default, use the reference for both jet selection and pt scaling
        jet_pt_for_selection = ref_jet.perp()
        jet_pt_for_scaling = ref_jet.perp()
        # if want to use leading parton instead of parton jet as ref_jet
        if self.matched_jet_type == 'p' and self.use_leading_parton:
            if jet.user_index()>0:
                jet_pt_for_selection = self.event[jet.user_index()].pT()
                jet_pt_for_scaling = self.event[jet.user_index()].pT()
            else:
                return # skip the histogram filling part if there is no valid ref_jet_pt (although technically, if ref_jet_pt is not used for selection nor scaling, one can still continue with the histogram filling. But we ignore this situation for now)
        if self.use_ref_for_jet_selection == False:
            jet_pt_for_selection = jet.perp()
        if self.use_ref_for_pt_scaling == False:
            jet_pt_for_scaling = jet.perp()

        for ipoint in range(2, self.npoint+1):
            for index in range(cb0.correlator(ipoint).rs().size()):
                    getattr(self, 'h_matched_ENC{}_JetPt_{}_R{}_trk00'.format(str(ipoint), level, R_label)).Fill(jet_pt_for_selection, cb0.correlator(ipoint).rs()[index], cb0.correlator(ipoint).weights()[index])
                    getattr(self, 'h_matched_ENC{}Pt_JetPt_{}_R{}_trk00'.format(str(ipoint), level, R_label)).Fill(jet_pt_for_selection, jet_pt_for_scaling*cb0.correlator(ipoint).rs()[index], cb0.correlator(ipoint).weights()[index])
            for index in range(cb1.correlator(ipoint).rs().size()):
                    getattr(self, 'h_matched_ENC{}_JetPt_{}_R{}_trk10'.format(str(ipoint), level, R_label)).Fill(jet_pt_for_selection, cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
                    getattr(self, 'h_matched_ENC{}Pt_JetPt_{}_R{}_trk10'.format(str(ipoint), level, R_label)).Fill(jet_pt_for_selection, jet_pt_for_scaling*cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])

        getattr(self, 'h_matched_Nconst_JetPt_{}_R{}_trk00'.format(level, R_label)).Fill(jet_pt_for_selection, len(_c_select0))
        getattr(self, 'h_matched_Nconst_JetPt_{}_R{}_trk10'.format(level, R_label)).Fill(jet_pt_for_selection, len(_c_select1))

    #---------------------------------------------------------------
    # Find jets, do matching between levels, and fill histograms & trees
    #---------------------------------------------------------------
    def find_jets_fill_trees(self):
        # Loop over jet radii
        for jetR in self.jetR_list:

            jetR_str = str(jetR).replace('.', '')
            jet_selector = getattr(self, "jet_selector_R%s" % jetR_str)
            jet_def = getattr(self, "jet_def_R%s" % jetR_str)
            track_selector_ch = getattr(self, "track_selector_ch")

            jets_p = fj.sorted_by_pt(jet_selector(jet_def(self.parts_pythia_p)))
            jets_h = fj.sorted_by_pt(jet_selector(jet_def(self.parts_pythia_h)))
            jets_ch = fj.sorted_by_pt(jet_selector(jet_def(track_selector_ch(self.parts_pythia_ch))))

            #-------------------------------------------------------------
            # match parton jets to the leading parton pdg id
            for jet_p in jets_p:
                matched_parton_parents = []
                for parton_parent in self.parton_parents:
                    if parton_parent.perp()/jet_p.perp() < 0.1:
                        continue
                    if parton_parent.perp()/jet_p.perp() > 10:
                        continue
                    if self.is_loose_geo_matched(jet_p, parton_parent, jetR): # NB: using a looser matching criteria for intial parton tagging
                        matched_parton_parents.append(parton_parent)
                    
                if len(matched_parton_parents)==1: # accept if there is one match only (NB: but may be used multiple times)
                    jet_p.set_user_index(matched_parton_parents[0].user_index()) # save particle index to user index
                    # print('matched parton jet R',jetR,'pt',jet_p.perp(),'phi',jet_p.phi(),'eta',jet_p.eta())
                    # print('matched leading parton',matched_parton_parents[0].user_index(),'pt',matched_parton_parents[0].perp(),'phi',matched_parton_parents[0].phi(),'eta',matched_parton_parents[0].eta())
                else:
                    jet_p.set_user_index(-1) # set user index to -1 fr no match case

                # print('all parton jet R',jetR,'pt',jet_p.perp(),'phi',jet_p.phi(),'eta',jet_p.eta(),'id',jet_p.user_index())

            R_label = str(jetR).replace('.', '') + 'Scaled'

            for jet_level in self.jet_levels:
                # Get the jets at different levels
                if jet_level == "p":
                    jets = jets_p
                if jet_level == "h":
                    jets = jets_h
                if jet_level == "ch":
                    jets = jets_ch

                #-------------------------------------------------------------
                # loop over jets and fill EEC histograms with jet constituents
                for j in jets:
                    self.fill_jet_histograms(jet_level, j, jetR, R_label)

                #-------------------------------------------------------------
                # loop over jets and fill EEC histograms inside a cone around jets
                if self.beyond_jetR and (jetR == self.ref_jetR) and (jet_level == self.ref_jet_level):
                    for j in jets:
                        for part_level in self.part_levels:
                            self.fill_beyond_jet_histograms(jet_level, part_level, j, jetR, R_label)
            
            if self.do_matching and (jetR == self.ref_jetR):
                # Loop through jets and find all h jets that can be matched to ch
                jets_h_matched_to_ch = []
                for jet_ch in jets_ch:
                    matched_jets_h = []
                    for index_jet_h, jet_h in enumerate(jets_h):
                        if jet_h.perp()/jet_ch.perp() < 0.1:
                            break
                        if jet_h.perp()/jet_ch.perp() > 10:
                            continue
                        if self.is_geo_matched(jet_ch, jet_h, jetR):
                            matched_jets_h.append(index_jet_h)
                    
                    if len(matched_jets_h)==1: # accept if there is one match only (NB: but mayb be used multiple times)
                        jets_h_matched_to_ch.append(matched_jets_h[0])
                    else:
                        jets_h_matched_to_ch.append(-1)

                # Loop through jets and find all p jets that can be matched to ch
                jets_p_matched_to_ch = []
                for jet_ch in jets_ch:
                    matched_jets_p = []
                    for index_jet_p, jet_p in enumerate(jets_p):
                        if jet_p.perp()/jet_ch.perp() < 0.1:
                            break
                        if jet_p.perp()/jet_ch.perp() > 10:
                            continue
                        if self.is_geo_matched(jet_ch, jet_p, jetR):
                            matched_jets_p.append(index_jet_p)
                    
                    if len(matched_jets_p)==1: # accept if there is one match only (NB: but mayb be used multiple times)
                        jets_p_matched_to_ch.append(matched_jets_p[0])
                    else:
                        jets_p_matched_to_ch.append(-1)

                #-------------------------------------------------------------
                # loop over matched jets and fill EEC histograms with jet constituents
                nmatched_ch = 0
                for index_j_ch, j_ch in enumerate(jets_ch):
                    imatched_p = jets_p_matched_to_ch[index_j_ch]
                    imatched_h = jets_h_matched_to_ch[index_j_ch]
                    if imatched_p > -1 and imatched_h > -1:
                        j_p = jets_p[imatched_p]
                        j_h = jets_h[imatched_h]
                        # print('matched ch',j_ch.perp(),'phi',j_ch.phi(),'eta',j_ch.eta())
                        # print('matched h',j_h.perp(),'phi',j_h.phi(),'eta',j_h.eta(),'dR',j_ch.delta_R(j_h))
                        # print('matched p',j_p.perp(),'phi',j_p.phi(),'eta',j_p.eta(),'dR',j_ch.delta_R(j_p))
                        nmatched_ch += 1

                        # used matched parton jet to tag the ch and h jet (qurak or gluon jet)
                        j_ch.set_user_index(j_p.user_index())
                        j_h.set_user_index(j_p.user_index())

                        if self.do_gluon_jet or self.do_quark_jet:
                            # skip the rest of processing on matched quark or gluon jets if the matched parton jet is not matched to a leading parton
                            if j_p.user_index() < 0:
                                continue
                            else:
                                leading_parton_id = abs(self.event[j_p.user_index()].id())
                                # if only want to process gluon jets but this jet is a quark jet, skip
                                if self.do_gluon_jet and (leading_parton_id>0 and leading_parton_id<7):
                                    continue
                                # if only want to process quark jets but this jet is a gluon jet, skip
                                if self.do_quark_jet and (leading_parton_id==9 or leading_parton_id==21):
                                    continue
                        
                        # fill histograms (using ch jet as reference) 
                        if self.matched_jet_type == 'ch':
                            self.fill_matched_jet_histograms('ch', j_ch, j_ch, R_label)
                            self.fill_matched_jet_histograms('p', j_p, j_ch, R_label)
                            self.fill_matched_jet_histograms('h', j_h, j_ch, R_label)

                         # fill histograms (using full jet as reference)
                        if self.matched_jet_type == 'h':
                            self.fill_matched_jet_histograms('ch', j_ch, j_h, R_label)
                            self.fill_matched_jet_histograms('p', j_p, j_h, R_label)
                            self.fill_matched_jet_histograms('h', j_h, j_h, R_label)

                        # fill histograms (using parton jet as reference)
                        # in this case, if use_leading_parton is enabled and no leading parton is matched to parton jet, fill_matched_jet_histograms() will not fill histograms
                        if self.matched_jet_type == 'p':
                            self.fill_matched_jet_histograms('ch', j_ch, j_p, R_label)
                            self.fill_matched_jet_histograms('p', j_p, j_p, R_label)
                            self.fill_matched_jet_histograms('h', j_h, j_p, R_label)

                        # fill histograms (using the same jets as reference)
                        if self.matched_jet_type == 'self':
                            self.fill_matched_jet_histograms('ch', j_ch, j_ch, R_label)
                            self.fill_matched_jet_histograms('p', j_p, j_p, R_label)
                            self.fill_matched_jet_histograms('h', j_h, j_h, R_label)

                        ref_parton_pt = j_p.perp()
                        # if want to use leading parton pt for reference and leading parton is matched to parton jet
                        if self.use_leading_parton:
                            if j_p.user_index() > 0:
                                leading_parton_pt = self.event[j_p.user_index()].pT()
                                ref_parton_pt = leading_parton_pt
                                hname = 'h_matched_JetPt_p_vs_p_R{}'.format(R_label)
                                getattr(self, hname).Fill(j_p.perp(), ref_parton_pt)
                                hname = 'h_matched_JetPt_p_over_p_ratio_R{}'.format(R_label)
                                getattr(self, hname).Fill(ref_parton_pt/j_p.perp(), j_p.perp())
                            else:
                                continue

                        hname = 'h_matched_JetPt_ch_vs_p_R{}'.format(R_label)
                        getattr(self, hname).Fill(j_ch.perp(), ref_parton_pt)
                        hname = 'h_matched_JetPt_h_vs_p_R{}'.format(R_label)
                        getattr(self, hname).Fill(j_h.perp(), ref_parton_pt)
                        hname = 'h_matched_JetPt_ch_vs_h_R{}'.format(R_label)
                        getattr(self, hname).Fill(j_ch.perp(), j_h.perp())

                        hname = 'h_matched_JetPt_p_over_ch_ratio_R{}'.format(R_label)
                        getattr(self, hname).Fill(ref_parton_pt/j_ch.perp(), j_ch.perp())
                        hname = 'h_matched_JetPt_p_over_h_ratio_R{}'.format(R_label)
                        getattr(self, hname).Fill(ref_parton_pt/j_h.perp(), j_h.perp())

                # if len(jets_ch)>0:
                #     print('matching efficiency:',nmatched_ch/len(jets_ch),'=',nmatched_ch,'/',len(jets_ch))
                # else:
                #     print('matching efficiency:',nmatched_ch,'/',len(jets_ch))
                    
    #---------------------------------------------------------------
    # Compare two jets and store matching candidates in user_info
    #---------------------------------------------------------------
    def is_geo_matched(self, jet1, jet2, jetR):
        deltaR = jet1.delta_R(jet2)
      
        # Add a matching candidate to the list if it is within the geometrical cut
        if deltaR < self.jet_matching_distance * jetR:
            return True
        else:
            return False

    def is_loose_geo_matched(self, jet1, jet2, jetR):
        deltaR = jet1.delta_R(jet2)
      
        # Add a matching candidate to the list if it is within the geometrical cut
        if deltaR < jetR:
            return True
        else:
            return False

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

    process = PythiaGenENC(config_file=args.config_file, output_dir=args.output_dir, args=args)
    process.pythia_parton_hadron(args)