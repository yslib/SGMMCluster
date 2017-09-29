#pragma warning(disable:4996)
#include <stdio.h>
#include "linktable.h"
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
#include <iostream>
#endif

#define MAX_CMD_LEN 256
#define MAX_DESC_LEN 256
#define CMD_MAX_ARGV_NUM 10
typedef struct Node
{
	tLinkTableNode * pNext;

	char  cmd[MAX_CMD_LEN];
	char  desc[MAX_DESC_LEN];
	int(*handler)(int argc,char *argv[]);
}tNode;
int Condition(tLinkTableNode * pNode, void * args)
{
	char * cmd = (char *)args;
	tNode * p = (tNode *)pNode;
	if (strcmp(p->cmd,cmd) == 0)
	{
		return 0;
	}
	return -1;
}
int ShowAllCmds(tLinkTable * menu)
{
	tNode * p = (tNode *)GetLinkTableHead(menu);
	while (p != NULL)
	{
		printf("%s -------- %s\n", p->cmd,p->desc);
		p = (tNode *)GetNextLinkTableNode(menu, (tLinkTableNode *)p);
	}
	return 0;
}
int quit(int argc, char ** argv)
{
	exit(0);
}
tNode * FindCmd(tLinkTable * menu, char * cmd)
{
	return (tNode *)SearchLinkTableNode(menu,Condition,cmd);
}
tLinkTable * menu;	//menu 
//
//build-in cmd handler
//
int help(int argc,char *argv[])
{
	return ShowAllCmds(menu);
}

int MenuConfig(char * cmd,char * desc,int (*handler)(int argc,char *argv[]))
{
	tNode* pNode = NULL;
	if (menu == NULL)
	{
		menu = CreateLinkTable();
		pNode = (tNode*)malloc(sizeof(tNode));
		if (pNode == NULL)
		{
			return 1;
		}
		strcpy(pNode->cmd, "help");
		strcpy(pNode->desc, "printf the help list");
		pNode->handler = help;
		AddLinkTableNode(menu, (tLinkTableNode *)pNode);
	}
	pNode = (tNode*)malloc(sizeof(tNode));
	strcpy(pNode->cmd, cmd);
	strcpy(pNode->desc,desc);
	pNode->handler = handler;
	AddLinkTableNode(menu, (tLinkTableNode *)pNode);
	return 0;
}
int ExecuteMenu()
{

	char  cmd[MAX_CMD_LEN];
	tNode * pCmdNode;
	ShowAllCmds(menu);

	char * pcmd;
	int argc;
	char * argv[CMD_MAX_ARGV_NUM];
	while (1)
	{
		printf(">>>");

		pcmd = fgets(cmd, MAX_CMD_LEN, stdin);
		//while (getchar() != '\n')continue;
		//empty the input buffer
		
		if (pcmd == NULL)
		{
			continue;
		}

		argc = 0;
		size_t len=strlen(pcmd);

		//change '\n' to '\0'
		pcmd[len-1]='\0';
		pcmd = strtok(pcmd, " ");
		//build parameter list
		while (pcmd != NULL && argc<CMD_MAX_ARGV_NUM)
		{
			argv[argc] = pcmd; 
			argc++;
			pcmd = strtok(NULL, " ");
		}
		pCmdNode = FindCmd(menu, cmd);
		if (pCmdNode == NULL)
		{
			printf("wrong commond.\n");
			continue;
		}
		if (pCmdNode->handler != NULL)
		{
			pCmdNode->handler(argc,argv);
			//WARNING:There may be potential BUG.
			//
			//while (getchar() != '\n')continue;
		}
	}
	return 0;
}
