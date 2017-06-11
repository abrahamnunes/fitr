
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# CONTACT INFO:
#   Abraham Nunes
#    Email: nunes@dal.ca
#
# ============================================================================

"""
Metrics used by various Fitr modules

Module Documentation
--------------------
"""
import numpy as np

def BIC(loglik, nparams, nsteps):
    """
    Calculates Bayesian information criterion

    Parameters
    ----------
    loglik : float or ndarray(dtype=float)
        Log-likelihood
    nparams : int
        Number of parameters in the model
    nsteps : int
        Number of time steps in the task

    Returns
    -------
    float or ndarray(dtype=float)

    """
    return nparams*np.log(nsteps) - 2*loglik

def AIC(nparams, loglik):
    """
    Calculates Aikake information criterion

    Parameters
    ----------
    nparams : int
        Number of parameters in the model
    loglik : float or ndarray(dtype=float)
        Log-likelihood

    Returns
    -------
    float or ndarray(dtype=float)

    """
    return 2*nparams - 2*loglik

def LME(logpost, nparams, hessian):
    """
    Laplace approximated log-model-evidence (LME)

    Parameters
    ----------
    logpost : float or ndarray(dtype=float)
        Log-posterior probability
    nparams : int
        Number of parameters in the model
    hessian : ndarray(size=(nparams, nparams))
        Hessian computed from parameter optimization

    Returns
    -------
    float or ndarray(dtype=float)
    """
    return logpost + (nparams/2)*np.log(2*np.pi)-np.log(np.linalg.det(hessian))/2
