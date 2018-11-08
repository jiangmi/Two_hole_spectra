"""Contains a class with the parameters of the hamiltonian."""
import numpy as np

class Parameters:
    """Specifies all the parameters of the hamiltonian. The cutoff Mc and
    the momentum k are not included. Nor is the energy omega. The lattice
    constant is set to 1.
    """
    def __init__(self, Norb, tpd, tpp, eps_d, eps_p, eta):
        self.Norb = Norb
        self.tpd = tpd 
        self.tpp = tpp 
        self.eps_d = eps_d # on-site energy for d orbitals
        self.eps_p = eps_p # on-site energy for p orbitals
        self.eta = eta # finite lifetime

    def print_all(self):
        print "Norb = {:.5f}".format(self.Norb)
        print "tpd = {:.5f}".format(self.tpd)
        print "tpp = {:.5f}".format(self.tpp)
        print "eps_d = {:.5f}".format(self.eps_d)
        print "eps_p = {:.5f}".format(self.eps_p)
        print "eta = {:.5f}".format(self.eta)

    def create_header(self):
        header = "#\tNorb = {:.5f}\n".format(self.Norb)
        header = "#\ttpd = {:.5f}\n".format(self.tpd)
        header += "#\ttpp = {:.5f}\n".format(self.tpp)
        header += "#\teps_d = {:.5f}\n".format(self.eps_d)
        header += "#\teps_p = {:.5f}\n".format(self.eps_p)
        header += "#\teta = {:.5f}".format(self.eta)

        return header

