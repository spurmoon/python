#include <Windows.h>
#include <QtCore\QString>
#include <QtCore\QLibrary>
#include <QtCore\QDir>

typedef wchar_t* (*TPy_DecodeLocale)(const char*, size_t*);
typedef void(*TPy_SetProgramName)(wchar_t*);
typedef void(*TPy_Initialize)();
typedef int(*TPyRun_SimpleString)(const char *);
typedef void(*TPy_Finalize)(void);
typedef void(*TPyMem_RawFree)(wchar_t*);
typedef int(*TPyRun_SimpleFile) (FILE*, const char*);
typedef FILE* (*T_Py_fopen)(const char *, const char *);
typedef int(*TPyRun_SimpleFileEx) (FILE *, const char *, int);

TPy_DecodeLocale Py_DecodeLocale;
TPy_SetProgramName Py_SetProgramName;
TPy_Initialize Py_Initialize;
TPyRun_SimpleString PyRun_SimpleString;
TPy_Finalize Py_Finalize;
TPyMem_RawFree PyMem_RawFree;
TPyRun_SimpleFile PyRun_SimpleFile;
T_Py_fopen _Py_fopen;
TPyRun_SimpleFileEx PyRun_SimpleFileEx;

#ifdef _DEBUG
#define HOME "E:/TEST/RiskAssessment"
#endif

int main(int argc, char* argv[])
{
#ifdef _DEBUG
    QString path = QString(HOME);
#else
    QString path = QDir::currentPath();
#endif
    SetDllDirectoryA(QString(path).append("/tools/").toLatin1());
    QLibrary* python = new QLibrary(QString(path).append("/tools/python35.dll").toLatin1());
    python->load();
    if (python->isLoaded()) {
        Py_DecodeLocale = (TPy_DecodeLocale)python->resolve("Py_DecodeLocale");
        Py_SetProgramName = (TPy_SetProgramName)python->resolve("Py_SetProgramName");
        Py_Initialize = (TPy_Initialize)python->resolve("Py_Initialize");
        PyRun_SimpleString = (TPyRun_SimpleString)python->resolve("PyRun_SimpleString");
        Py_Finalize = (TPy_Finalize)python->resolve("Py_Finalize");
        PyMem_RawFree = (TPyMem_RawFree)python->resolve("PyMem_RawFree");
        PyRun_SimpleFile = (TPyRun_SimpleFile)python->resolve("PyRun_SimpleFileEx");
        _Py_fopen = (T_Py_fopen)python->resolve("_Py_fopen");
        PyRun_SimpleFileEx = (TPyRun_SimpleFileEx)python->resolve("PyRun_SimpleFileEx");
    }

    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    Py_SetProgramName(program);  /* optional but recommended */
    Py_Initialize();
    FILE* dotpy = _Py_fopen(QString(path).append("/tools/opti.py").toLatin1(), "r");
    PyRun_SimpleFileEx(dotpy, "opti.py", 1);
    Py_Finalize();
    PyMem_RawFree(program);
    getchar();
    return 0;
}
