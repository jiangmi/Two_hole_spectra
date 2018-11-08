"""Contains a class with the parameters of the hamiltonian."""
import numpy as np

class Parameters:
    """Specifies all the parameters of the hamiltonian. The cutoff Mc and
    the momentum k are not included. Nor is the energy omega. The lattice
    constant is set to 1.
    """
    def __init__(self, pds, pdp, pps, ppp, eps_d, eps_p, eta):
        self.pds = pds
        self.pdp = pdp 
        self.pps = pps 
        self.ppp = ppp
        self.eps_d = eps_d # on-site energy for d orbitals
        self.eps_p = eps_p # on-site energy for p orbitals
        self.eta = eta # finite lifetime

    def print_all(self):
        print "pds = {:.5f}".format(self.pds)
        print "pdp = {:.5f}".format(self.pdp)
        print "pps = {:.5f}".format(self.pps)
        print "ppp = {:.5f}".format(self.ppp)
        print "eps_d = {:.5f}".format(self.eps_d)
        print "eps_p = {:.5f}".format(self.eps_p)
        print "eta = {:.5f}".format(self.eta)

    def create_header(self):
        header = "#\tpds = {:.5f}\n".format(self.pds)
        header = "#\tpdp = {:.5f}\n".format(self.pdp)
        header = "#\tpps = {:.5f}\n".format(self.pps)
        header = "#\tppp = {:.5f}\n".format(self.ppp)
        header += "#\teps_d = {:.5f}\n".format(self.eps_d)
        header += "#\teps_p = {:.5f}\n".format(self.eps_p)
        header += "#\teta = {:.5f}".format(self.eta)

        return header

