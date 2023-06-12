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

        self.jetR_list = config["jetR"] 

        self.nev = args.nev

        # particle level - ALICE tracking restriction
        self.max_eta_hadron = 0.9

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

            levels = ["p", "h", "ch"]

            for level in levels:
                # ENC histograms (trk_thrd = 1)
                for ipoint in range(2, 3):
                    name = 'h_ENC{}_JetPt_{}_R{}_trk00'.format(str(ipoint), level, R_label)
                    print('Initialize histogram',name)
                    pt_bins = linbins(0,200,200)
                    RL_bins = logbins(1E-4,1,50)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('pT (full jet)')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    name = 'h_ENC{}_JetPt_{}_R{}_trk10'.format(str(ipoint), level, R_label)
                    print('Initialize histogram',name)
                    pt_bins = linbins(0,200,200)
                    RL_bins = logbins(1E-4,1,50)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('pT (full jet)')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    # only save charge separation for pT>1GeV for now
                    if level == "ch":
                        name = 'h_ENC{}_JetPt_{}_R{}_unlike_trk10'.format(str(ipoint), level, R_label)
                        print('Initialize histogram',name)
                        pt_bins = linbins(0,200,200)
                        RL_bins = logbins(1E-4,1,50)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('pT (full jet)')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        name = 'h_ENC{}_JetPt_{}_R{}_like_trk10'.format(str(ipoint), level, R_label)
                        print('Initialize histogram',name)
                        pt_bins = linbins(0,200,200)
                        RL_bins = logbins(1E-4,1,50)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('pT (full jet)')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                # Jet pt vs N constituents
                name = 'h_Nconst_JetPt_{}_R{}_trk00'.format(level, R_label)
                print('Initialize histogram',name)
                pt_bins = linbins(0,200,200)
                Nconst_bins = linbins(0,50,50)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, Nconst_bins)
                h.GetXaxis().SetTitle('pT (full jet)')
                h.GetYaxis().SetTitle('N_{const}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                name = 'h_Nconst_JetPt_{}_R{}_trk10'.format(level, R_label)
                print('Initialize histogram',name)
                pt_bins = linbins(0,200,200)
                Nconst_bins = linbins(0,50,50)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, Nconst_bins)
                h.GetXaxis().SetTitle('pT (full jet)')
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

        pfc_selector1 = fj.SelectorPtMin(1.)
        setattr(self, "pfc_def_10", pfc_selector1)

        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')
            
            jet_selector = fj.SelectorPtMin(0.15) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)
            setattr(self, "jet_selector_R%s" % jetR_str, jet_selector)

    #---------------------------------------------------------------
    # Calculate events and pass information on to jet finding
    #---------------------------------------------------------------
    def calculate_events(self, pythia):
        
        iev = 0  # Event loop count

        print('ievt',iev)

        while iev < self.nev:
            if not pythia.next():
                continue

            self.event = pythia.event

            parts_pythia_p = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True) # final stable partons

            hstatus = pythia.forceHadronLevel()
            if not hstatus:
                continue

            # full particle level
            parts_pythia_h = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True)

            # charged particle level
            parts_pythia_ch = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)

            # Some "accepted" events don't survive hadronization step -- keep track here
            self.hNevents.Fill(0)
            self.find_jets_fill_trees('p', parts_pythia_p, iev)
            self.find_jets_fill_trees('h', parts_pythia_h, iev)
            self.find_jets_fill_trees('ch', parts_pythia_ch, iev)

            iev += 1

    #---------------------------------------------------------------
    # Find jets, do matching between levels, and fill histograms & trees
    #---------------------------------------------------------------
    def find_jets_fill_trees(self, level, parts_pythia, iev):
        # Loop over jet radii
        for jetR in self.jetR_list:

            jetR_str = str(jetR).replace('.', '')
            jet_selector = getattr(self, "jet_selector_R%s" % jetR_str)
            jet_def = getattr(self, "jet_def_R%s" % jetR_str)
            
            pfc_selector1 = getattr(self, "pfc_def_10")
 
            # Get the jets at different levels
            jets  = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia)))

            R_label = str(jetR).replace('.', '') + 'Scaled'

            #-------------------------------------------------
            # loop over jets
            for j in jets:

                dphi_cut = -9999
                deta_cut = -9999
                npoint = 2
                npower = 1

                # select all constituents with no cut
                _c_select0 = fj.vectorPJ()
                _ = [_c_select0.push_back(c) for c in j.constituents()]
                cb0 = ecorrel.CorrelatorBuilder(_c_select0, j.perp(), npoint, npower, dphi_cut, deta_cut)
       
                # select constituents with 1 GeV cut
                _c_select1 = fj.vectorPJ()
                _ = [_c_select1.push_back(c) for c in pfc_selector1(j.constituents())]
                cb1 = ecorrel.CorrelatorBuilder(_c_select1, j.perp(), npoint, npower, dphi_cut, deta_cut)

                for ipoint in range(2, npoint+1):
                    for index in range(cb0.correlator(ipoint).rs().size()):
                            getattr(self, 'h_ENC{}_JetPt_{}_R{}_trk00'.format(str(ipoint), level, R_label)).Fill(j.perp(), cb0.correlator(ipoint).rs()[index], cb0.correlator(ipoint).weights()[index])
                    for index in range(cb1.correlator(ipoint).rs().size()):
                            getattr(self, 'h_ENC{}_JetPt_{}_R{}_trk10'.format(str(ipoint), level, R_label)).Fill(j.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
                    
                if level == "ch":
                    for ipoint in range(2, npoint+1):
                        # only fill trk pt > 1 GeV here for now
                        for index in range(cb1.correlator(ipoint).rs().size()):
                            part1 = cb1.correlator(ipoint).indices1()[index]
                            part2 = cb1.correlator(ipoint).indices2()[index]
                            c1 = _c_select1[part1]
                            c2 = _c_select1[part2]
                            if pythiafjext.getPythia8Particle(c1).charge()*pythiafjext.getPythia8Particle(c2).charge() < 0:
                                # print("unlike-sign pair ",pythiafjext.getPythia8Particle(c1).id(),pythiafjext.getPythia8Particle(c2).id())
                                getattr(self, 'h_ENC{}_JetPt_{}_R{}_unlike_trk10'.format(str(ipoint), level, R_label)).Fill(j.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
                            else:
                                # print("likesign pair ",pythiafjext.getPythia8Particle(c1).id(),pythiafjext.getPythia8Particle(c2).id())
                                getattr(self, 'h_ENC{}_JetPt_{}_R{}_like_trk10'.format(str(ipoint), level, R_label)).Fill(j.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
                
                getattr(self, 'h_Nconst_JetPt_{}_R{}_trk00'.format(level, R_label)).Fill(j.perp(), len(_c_select0))
                getattr(self, 'h_Nconst_JetPt_{}_R{}_trk10'.format(level, R_label)).Fill(j.perp(), len(_c_select1))
        
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