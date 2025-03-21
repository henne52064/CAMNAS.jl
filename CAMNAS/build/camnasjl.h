#include <dpsim/MNASolverDynInterface.h>

int init(struct dpsim_csr_matrix *matrix);
int decomp(struct dpsim_csr_matrix *matrix);
int solve(double *rhs_values, double *lhs_values);
void log(const char *str);
void cleanup(void);