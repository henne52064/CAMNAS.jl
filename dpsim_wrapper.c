#include <stdio.h>
#include <string.h>

#include <julia_init.h>
#include <camnasjl.h>

#include <dpsim/MNASolverDynInterface.h>

void __attribute__((constructor)) init_lib (void){
    printf("This is the library constructor...\n");
    printf("Initializing Julia...\n");
    printf("==============================\n");
    char** argv;
    int argc = 0;

    init_julia(argc, argv);

}

void __attribute__((destructor)) shutdown_lib (void){
    printf("This is the library destructor...\n");
    printf("Shutting down Julia...\n");
    printf("==============================\n");
    shutdown_julia(0);
}

static const char* PLUGIN_NAME = "camnasjl";
static struct dpsim_mna_plugin solver_plugin = {
	.log = log,
	.init = init,
	.lu_decomp = decomp,
	.solve = solve,
	.cleanup = cleanup,
};

struct dpsim_mna_plugin *get_mna_plugin(const char *name) {
    if (name == NULL || strcmp(name, PLUGIN_NAME) != 0) {
        printf("error: name mismatch %s %s\n", name, PLUGIN_NAME);
        return NULL;
    }
    return &solver_plugin;
}

