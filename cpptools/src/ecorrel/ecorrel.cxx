#include "ecorrel.hh"
#include <cmath>
#include <stdexcept>
#include <fastjet/Selector.hh>

namespace EnergyCorrelators
{
    CorrelatorsContainer::CorrelatorsContainer()
    : fr()
    , fw()
    , frxw()
    , findx1()
    , findx2()
    {
        ;
    }
    
    CorrelatorsContainer::~CorrelatorsContainer()
    {
        ;
    }

    void CorrelatorsContainer::clear()
    {
        fw.clear();
        fr.clear();
        frxw.clear();
    }

    void CorrelatorsContainer::addwr(const double &w, const double &r)
    {
        fw.push_back(w);
        fr.push_back(r);
    }

    void CorrelatorsContainer::addwr(const double &w, const double &r, const int &indx1, const int &indx2)
    {
        fw.push_back(w);
        fr.push_back(r);
        findx1.push_back(indx1);
        findx2.push_back(indx2);
    }

    std::vector<double> *CorrelatorsContainer::weights()
    {
        return &fw;
    }

    std::vector<double> *CorrelatorsContainer::rs()
    {
        return &fr;
    }

    std::vector<int> *CorrelatorsContainer::indices1()
    {
        return &findx1;
    }

    std::vector<int> *CorrelatorsContainer::indices2()
    {
        return &findx2;
    }

    const double *CorrelatorsContainer::wa()
    {
        return &fw[0];
    }

    const double *CorrelatorsContainer::ra()
    {
        return &fr[0];
    }

    std::vector<double> *CorrelatorsContainer::rxw()
    {
        frxw.clear();
        for (size_t i = 0; i < fr.size(); i++)
        {
            frxw.push_back(fr[i] * fw[i]);
        }
        return &frxw;
    }

    std::vector<fastjet::PseudoJet> constituents_as_vector(const fastjet::PseudoJet &jet)
    {
        std::vector<fastjet::PseudoJet> _v;
        for (auto &c : jet.constituents())
        {
            _v.push_back(c);
        }
        return _v;
    }

    CorrelatorBuilder::CorrelatorBuilder()
    : fec()
    , fncmax(0)
    {
        ;
    }

    CorrelatorBuilder::CorrelatorBuilder(const std::vector<fastjet::PseudoJet> &parts, const double &scale, const int &nmax, const int &power, const double dphi_cut = -9999, const double deta_cut = -9999)
    : fec()
    , fncmax(nmax)
    {
        // std::cout << "Initializing n point correlator with power " << power << " for " << parts.size() << " paritlces" << std::endl;
        if (fncmax < 2)
        {
            throw std::overflow_error("asking for n-point correlator with n < 2?");
        }
        if (fncmax > 5)
        {
            throw std::overflow_error("max n for n-point correlator is currently 4");
        }
        for (int i = 0; i < fncmax - 2 + 1; i++)
        {
            fec.push_back(new CorrelatorsContainer());
        }
        for (size_t i = 0; i < parts.size(); i++)
        {
            for (size_t j = 0; j < parts.size(); j++)
            {
                double _phi12 = fabs(parts[i].delta_phi_to(parts[j])); // expecting delta_phi_to() to return values in [-pi, pi]
                double _eta12 = parts[i].eta() - parts[j].eta();
                if (dphi_cut > -1)
                { // if dphi_cut is on, apply it to pairs
                    double _pt1 = parts[i].pt();
                    double _pt2 = parts[j].pt();
                    int _q1 = 1; // FIX ME: just dummy (no charge info available yet in data and full sim)
                    int _q2 = 1;
                    if ( !ApplyDeltaPhiRejection(dphi_cut, _q1, _q2, _pt1, _pt2, _phi12) ) continue;
                }
                if (deta_cut > -1)
                { // if deta_cut is on, apply it to pairs
                    if ( !ApplyDeltaEtaRejection(deta_cut, _eta12) ) continue;
                }
                double _d12 = parts[i].delta_R(parts[j]);
                double _w2 = parts[i].perp() * parts[j].perp() / std::pow(scale, 2);
                _w2 = pow(_w2, power);
                fec[2 - 2]->addwr(_w2, _d12, i, j); // save weight, distance and indices of the pair
                if (fncmax < 3)
                    continue;
                for (size_t k = 0; k < parts.size(); k++)
                {
                    double _d13 = parts[i].delta_R(parts[k]);
                    double _d23 = parts[j].delta_R(parts[k]);
                    double _w3 = parts[i].perp() * parts[j].perp() * parts[k].perp() / std::pow(scale, 3);
                    _w3 = pow(_w3, power);
                    double _d3max = std::max({_d12, _d13, _d23});
                    if (fabs(_d3max-_d12)<1E-5) fec[3 - 2]->addwr(_w3, _d3max, i, j);
                    if (fabs(_d3max-_d13)<1E-5) fec[3 - 2]->addwr(_w3, _d3max, i, k);
                    if (fabs(_d3max-_d23)<1E-5) fec[3 - 2]->addwr(_w3, _d3max, j, k);
                    if (fncmax < 4)
                        continue;
                    for (size_t l = 0; l < parts.size(); l++)
                    {
                        double _d14 = parts[i].delta_R(parts[l]);
                        double _d24 = parts[j].delta_R(parts[l]);
                        double _d34 = parts[k].delta_R(parts[l]);
                        double _w4 = parts[i].perp() * parts[j].perp() * parts[k].perp() * parts[l].perp() / std::pow(scale, 4);
                        _w4 = pow(_w4, power);
                        double _d4max = std::max({_d12, _d13, _d23, _d14, _d24, _d34});
                        if (fabs(_d4max-_d12)<1E-5) fec[4 - 2]->addwr(_w4, _d4max, i, j);
                        if (fabs(_d4max-_d13)<1E-5) fec[4 - 2]->addwr(_w4, _d4max, i, k);
                        if (fabs(_d4max-_d23)<1E-5) fec[4 - 2]->addwr(_w4, _d4max, j, k);
                        if (fabs(_d4max-_d14)<1E-5) fec[4 - 2]->addwr(_w4, _d4max, i, l);
                        if (fabs(_d4max-_d24)<1E-5) fec[4 - 2]->addwr(_w4, _d4max, j, l);
                        if (fabs(_d4max-_d34)<1E-5) fec[4 - 2]->addwr(_w4, _d4max, k, l);
                        if (fncmax < 5)
                            continue;
                        for (size_t m = 0; m < parts.size(); m++)
                        {
                            double _d15 = parts[i].delta_R(parts[m]);
                            double _d25 = parts[j].delta_R(parts[m]);
                            double _d35 = parts[k].delta_R(parts[m]);
                            double _d45 = parts[l].delta_R(parts[m]);
                            double _w5 = parts[i].perp() * parts[j].perp() * parts[k].perp() * parts[l].perp() * parts[m].perp() / std::pow(scale, 5);
                            _w5 = pow(_w5, power);
                            double _d5max = std::max({_d12, _d13, _d23, _d14, _d24, _d34, _d15, _d25, _d35, _d45});
                            fec[5 - 2]->addwr(_w5, _d5max); // the indices not filled for 5-point yet
                        }
                    }
                }
            }
        }
    }

    CorrelatorsContainer* CorrelatorBuilder::correlator(int n)
    {
        if (n > fncmax)
        {
            throw std::overflow_error("requesting n-point correlator with too large n");
        }
        if (n < 2)
        {
            throw std::overflow_error("requesting n-point correlator with n < 2?");
        }
        return fec[n - 2];
    }

    CorrelatorBuilder::~CorrelatorBuilder()
    {
        for (auto p : fec)
        {
            delete p;
        }
        fec.clear();
    }

    bool CorrelatorBuilder::ApplyDeltaPhiRejection(const double dphi_cut, const double q1, const double q2, const double pt1, const double pt2, const double phi12)
    {
        double R = 1.1; // reference radius for TPC
        double Bz = 0.5;
        double phi_star = phi12 + q1*asin(-0.015*Bz*R/pt1) - q2*asin(-0.015*Bz*R/pt2);
        if ( fabs(phi_star)<dphi_cut ) return false;  
        return true;
    }

    bool CorrelatorBuilder::ApplyDeltaEtaRejection(const double deta_cut, const double eta12)
    {
        if ( fabs(eta12)<deta_cut ) return false;
        return true;
    }

	std::vector<fastjet::PseudoJet> merge_signal_background_pjvectors(const std::vector<fastjet::PseudoJet> &signal, 
																	  const std::vector<fastjet::PseudoJet> &background,
																      const double pTcut,
																	  const int bg_index_start)
    {
        std::vector<fastjet::PseudoJet> _vreturn;
        auto _selector = fastjet::SelectorPtMin(pTcut);
        auto _signal = _selector(signal);
        for (auto &_p : _signal)
        {
            _p.set_user_index(_p.user_index());
            _vreturn.push_back(_p);
        }
        auto _background = _selector(background);
        for (auto &_p : _background)
        {
            if (bg_index_start > 0)
            {
                int _index = &_p - &_background[0];
                _p.set_user_index(bg_index_start + _index);
            }
            else
            {
                _p.set_user_index(_p.user_index());
            }
            _vreturn.push_back(_p);
        }
        return _vreturn;
    }

}
