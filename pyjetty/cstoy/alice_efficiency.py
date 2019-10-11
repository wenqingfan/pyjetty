#!/usr/bin/env python

from __future__ import print_function

import fastjet as fj

import tqdm
import argparse
import os
import numpy as np

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext
import pythiaext

import numpy as np
import array
import copy
import random

from pyjetty.mputils import logbins
from pyjetty.mputils import MPBase

import ROOT
ROOT.gROOT.SetBatch(True)

class AliceChargedParticleEfficiency(MPBase):
	def __init__(self, **kwargs):
		self.configure_from_args(csystem='pp')
		super(AliceChargedParticleEfficiency, self).__init__(**kwargs)
		if self.csystem == 'pp':
			self.effi_1GeV = 0.73
			self.effi_1GeVup = 0.83
		if self.csystem == 'PbPb':
			self.effi_1GeV = 0.73 - 0.02
			self.effi_1GeVup = 0.83 - 0.02

	def apply_efficiency(self, particles):
		output = []
		for p in particles:
			if p.pt() < 0.15:
				continue
			if p.pt() < 1:
				if random.random() > self.effi_1GeV:
					continue
				else:
					output.append(p)
			if p.pt() >= 1:
				if random.random() > self.effi_1GeVup:
					continue
				else:
					output.append(p)
		return output



def main():
	parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly', prog=os.path.basename(__file__))
	pyconf.add_standard_pythia_args(parser)
	parser.add_argument('--output', default="output.root", type=str)
	parser.add_argument('--overwrite', help="overwrite output", default=False, action='store_true')
	args = parser.parse_args()

	if args.output == 'output.root':
		args.output = 'output_alice_efficiency.root'

	if os.path.isfile(args.output):
		if not args.overwrite:
			print('[i] output', args.output, 'exists - use --overwrite to do just that...')
			return

	print(args)

	if args.py_seed >= 0:
		mycfg.append('Random:setSeed=on')
		mycfg.append('Random:seed={}'.format(args.py_seed))

	mycfg = []
	pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)
	if not pythia:
		print("[e] pythia initialization failed.")
		return

	max_eta = 1
	parts_selector = fj.SelectorAbsEtaMax(max_eta)

	if args.nev < 1:
		args.nev = 1

	outf = ROOT.TFile(args.output, 'recreate')
	outf.cd()
	hpart_gen   = ROOT.TH1F('hpart_gen', 'hpart gen;p_{T};counts', 20, logbins(0.01, 100, 20))
	heffi_pp 	= ROOT.TH1F('heffi_pp', 'heffi pp;p_{T};efficiency', 20, logbins(0.01, 100, 20))
	heffi_PbPb 	= ROOT.TH1F('heffi_PbPb', 'heffi PbPb;p_{T};efficiency', 20, logbins(0.01, 100, 20))

	effi_pp = AliceChargedParticleEfficiency(csystem='pp')
	effi_PbPb = AliceChargedParticleEfficiency(csystem='PbPb')

	for iev in tqdm.tqdm(range(args.nev)):
		if not pythia.next():
			continue

		parts_pythia = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged])
		parts = parts_selector(parts_pythia)
		_tmp = [hpart_gen.Fill(p.pt()) for p in parts]
		_tmp = [heffi_pp.Fill(p.pt()) for p in effi_pp.apply_efficiency(parts)]
		_tmp = [heffi_PbPb.Fill(p.pt()) for p in effi_PbPb.apply_efficiency(parts)]
		continue

	pythia.stat()
	heffi_pp.Sumw2()
	heffi_pp.Divide(hpart_gen)
	heffi_PbPb.Sumw2()
	heffi_PbPb.Divide(hpart_gen)
	outf.Write()
	outf.Close()
	print('[i] written', outf.GetName())

if __name__ == '__main__':
	main()