#include "linktable.h"
#include <stdlib.h>
#include <stdio.h>
typedef struct LinkTableNode
{
	struct LinkTableNode * pNext;
}tLinkTableNode;
/*
*LinkTable Type
*/
typedef struct LinkTable
{
	tLinkTableNode * pHead;
	tLinkTableNode * pTail;
	int       SumOfNode;
}tLinkTable;
//************************************
// Method:    CreateLinkTable
// FullName:  CreateLinkTable
// Access:    public 
// Returns:   tLinkTable *
// Qualifier:
//************************************
tLinkTable * CreateLinkTable(void)
{
	tLinkTable * pNewLinkTable;
	pNewLinkTable = (tLinkTable *)malloc(sizeof(tLinkTable));

	if (pNewLinkTable == NULL)
	{
		printf("create linktable failed\n");
		exit(0);
	}
	pNewLinkTable->SumOfNode = 0;
	pNewLinkTable->pHead = pNewLinkTable->pTail = NULL;
	return pNewLinkTable;
}
//************************************
// Method:    DeleteLinkTable
// FullName:  DeleteLinkTable
// Access:    public 
// Returns:   int
// Qualifier:
// Parameter: tLinkTable * pLinkTable
//************************************
int DeleteLinkTable(tLinkTable * pLinkTable)
{
	tLinkTableNode * p;
	p = GetLinkTableHead(pLinkTable);
	//free the content of the linktable except the tail
	while (p != pLinkTable->pTail)
	{
		free(p);
		p = GetNextLinkTableNode(pLinkTable, p);
	}
	free(pLinkTable->pTail);
	//free the tLinkTable
	free(pLinkTable);
	return 1;
}
//************************************
// Method:    AddLinkTableNode
// FullName:  AddLinkTableNode
// Access:    public 
// Returns:   int
// Qualifier:
// Parameter: tLinkTable * pLinkTable
// Parameter: tLinkTableNode * pNode
//************************************
int AddLinkTableNode(tLinkTable * pLinkTable, tLinkTableNode * pNode)
{
	tLinkTableNode * pNewNode = pNode;
	tLinkTableNode * pHead = GetLinkTableHead(pLinkTable);
	if (pNewNode == NULL||pLinkTable == NULL)
	{
		return -1;//allocated for the new node failed
	}

	pNewNode->pNext = NULL;
	if (pHead != NULL) //non-empty
	{
		pLinkTable->pTail->pNext = pNewNode;
		pLinkTable->pTail = pNewNode;
	}
	else
	{	
		pLinkTable->pHead = pNewNode;
		pLinkTable->pTail = pNewNode;
	}
	pLinkTable->SumOfNode++;
	return 1;
}
//************************************
// Method:    DeleteLinkTableNode
// FullName:  DeleteLinkTableNode
// Access:    public 
// Returns:   int
// Qualifier:
// Parameter: tLinkTable * pLinkTable
// Parameter: tLinkTableNode * pNode
//************************************
int DeleteLinkTableNode(tLinkTable * pLinkTable, tLinkTableNode * pNode)
{
	tLinkTableNode * pNext, *pCurrent;
	tLinkTableNode * pToFree = NULL;
	pCurrent = GetLinkTableHead(pLinkTable);
	//linktable is empty
	if (pLinkTable == NULL || pCurrent == NULL)
	{
		return -1;
	}
	else
	{	
		pNext = GetNextLinkTableNode(pLinkTable, pCurrent);
		if (pCurrent == pNode)
		{
			pLinkTable->pHead = pNext->pNext;
			pToFree = pCurrent;
		}
		else
		{
			pNext = GetNextLinkTableNode(pLinkTable, pCurrent);
			while (pNext != pNode && pNext != NULL)
			{
				pCurrent = pNext;
				pNext = GetNextLinkTableNode(pLinkTable, pCurrent);
			}
			if (pNext != NULL)
			{
				pToFree = pNext;
				pCurrent->pNext = pNext->pNext;
			}
		}
	}
	free(pToFree);
	pLinkTable->SumOfNode--;
	return 1;
}

tLinkTableNode * SearchLinkTableNode(tLinkTable * pLinkTable, int(*condition)(tLinkTableNode * pNode, void * args), void * args)
{
	if (pLinkTable == NULL || condition == NULL)
	{
		return NULL;
	}
	tLinkTableNode * p = GetLinkTableHead(pLinkTable);
	while (p != NULL)
	{
		if (condition(p, args) == 0)
		{
			return p;
		}
		p = GetNextLinkTableNode(pLinkTable, p);
	}
	return NULL;
}

tLinkTableNode * GetLinkTableHead(tLinkTable *pLinkTable)
{
	return pLinkTable->pHead;
}
tLinkTableNode * GetNextLinkTableNode(tLinkTable *pLinkTable, tLinkTableNode * pNode)
{
	return pNode->pNext;
}