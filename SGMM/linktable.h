#pragma once
#ifndef _LINK_TABLE_E_
#define _LINK_TABLE_H_


#ifdef __cplusplus
extern "C" {
#endif

	typedef struct LinkTableNode tLinkTableNode;
	/*
	*LinkTable Type
	*/
	typedef struct LinkTable tLinkTable;
	/*
	*Create a LinkTable
	*/
	tLinkTable * CreateLinkTable();
	/*
	*Delete a LinkTable
	*/
	int DeleteLinkTable(tLinkTable * pLinkTable);
	/*
	*
	*/
	int AddLinkTableNode(tLinkTable * pLinkTable, tLinkTableNode * pNode);
	/*
	*
	*/
	int DeleteLinkTableNode(tLinkTable * pLinkTable, tLinkTableNode * pNode);
	/*
	*
	*/
	tLinkTableNode * SearchLinkTableNode(tLinkTable * pLinkTable, int(*condition)(tLinkTableNode * pNode, void * args), void * args);
	/*
	*
	*/

	tLinkTableNode * GetLinkTableHead(tLinkTable *pLinkTable);
	/*
	*
	*/
	tLinkTableNode * GetNextLinkTableNode(tLinkTable *pLinkTable, tLinkTableNode * pNode);


#ifdef __cplusplus
}
#endif


#endif
