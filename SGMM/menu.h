#pragma once
#ifndef _MENU_H_
#define _MENU_H_

#ifdef __cplusplus
extern "C" {
#endif
	/*add cmd to menu*/


	int MenuConfig(char * cmd, char * desc, int(*handler)(int argc, char *argv[]));

	/*Menu Engine Execute*/
	int ExecuteMenu();

	/*Show all commands*/
	int help(int argc, char *argv[]);
	/*quit from the program*/
	int quit(int argc, char ** argv);

	/**/


#ifdef __cplusplus
}
#endif


#endif	/* _MENU_H_*/