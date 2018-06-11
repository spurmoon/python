#include "stdafx.h"
#include "uishell.h"
#include <QMessageBox>
#include <QFile>

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

UIShell::UIShell(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    QAction* action_about = ui.menuBar->addAction("&About");

    qstring_Peri = ui.label_Ecc_Peri->text();
    qstring_Apo = ui.label_SMA_Apo->text();
    qstring_SMA = QString::fromLocal8Bit("半长轴(Km:)");
    qstring_Ecc = QString::fromLocal8Bit("偏心率：");
    //
    connect(action_about, &QAction::triggered, this,
        [this]() {
        QMessageBox::about(this, QString::fromLocal8Bit("关于"), QString::fromLocal8Bit("Need to be filled"));
        });
    connect(ui.action_Exit, &QAction::triggered, this, &QWidget::close);

    //
#ifdef _DEBUG
    QString path = QString(HOME);
#else
    QString path = QCoreApplication::applicationDirPath();
#endif
    SetDllDirectoryA(QString(path).append("/tools/").toLatin1());
    QLibrary* python = new QLibrary(QString(path).append("/tools/python35.dll").toLatin1(), this);
    python->load();    
    if (python->isLoaded()){
        Py_DecodeLocale = (TPy_DecodeLocale) python->resolve("Py_DecodeLocale");
        Py_SetProgramName = (TPy_SetProgramName) python->resolve("Py_SetProgramName");
        Py_Initialize = (TPy_Initialize) python->resolve("Py_Initialize");
        PyRun_SimpleString = (TPyRun_SimpleString) python->resolve("PyRun_SimpleString");
        Py_Finalize = (TPy_Finalize) python->resolve("Py_Finalize");
        PyMem_RawFree = (TPyMem_RawFree) python->resolve("PyMem_RawFree");
        PyRun_SimpleFile = (TPyRun_SimpleFile) python->resolve("PyRun_SimpleFileEx");
        _Py_fopen = (T_Py_fopen) python->resolve("_Py_fopen");
        PyRun_SimpleFileEx = (TPyRun_SimpleFileEx) python->resolve("PyRun_SimpleFileEx");
    }
    else{
        ui.statusBar->showMessage(python->errorString());
    }
}

UIShell::~UIShell()
{

}

void UIShell::sma_Ecc_fromto_Apo_Peri(int state)
{
    double sma, ecc, apo, peri;
    if (state == 0) {
        sma = ui.lineEdit_SMA_Apo->text().toDouble();
        ecc = ui.lineEdit_Ecc_Peri->text().toDouble();
        apo = sma*(1 + ecc) - 6378.;
        peri = sma*(1 - ecc) - 6378.;
        ui.label_SMA_Apo->setText(qstring_Apo);
        ui.label_Ecc_Peri->setText(qstring_Peri);
        ui.lineEdit_SMA_Apo->setText(QString::number(apo, 'f', 0));
        ui.lineEdit_Ecc_Peri->setText(QString::number(peri, 'f', 0));
    }
    else {
        apo = ui.lineEdit_SMA_Apo->text().toDouble();
        peri = ui.lineEdit_Ecc_Peri->text().toDouble();
        sma = (apo + peri) / 2. + 6378.;
        ecc = (apo + 6378.) / sma - 1.;
        ui.label_SMA_Apo->setText(qstring_SMA);
        ui.label_Ecc_Peri->setText(qstring_Ecc);
        ui.lineEdit_SMA_Apo->setText(QString::number(sma, 'f', 0));
        ui.lineEdit_Ecc_Peri->setText(QString::number(ecc, 'f', 3));
    }

    return;
}

void UIShell::sdModelTest()
{
    ui.statusBar->clearMessage();

    QProcess ordem2k;
#ifdef _DEBUG
    QString path = QString(HOME);
#else
    QString path = QCoreApplication::applicationDirPath();
#endif
    if(QFile::exists(path + "/tools/ORDEM2000.res")) {
        QFile::remove(path + "/tools/ORDEM2000.res");
    }
    ui.statusBar->showMessage(QString::fromLocal8Bit("模型测算1运行中..."));
    ordem2k.setWorkingDirectory(path + "/tools/");
    ordem2k.setProgram(path + "/tools/ORDEM2K.exe");
    ordem2k.start();
    if ( ordem2k.waitForFinished() && QFile::exists(path + "/tools/ORDEM2000.res") ){
        ui.statusBar->showMessage(QString::fromLocal8Bit("模型测算2运行中..."));
    }
    else{
        ui.statusBar->showMessage("ordem Error...");
    }
    ordem2k.setProgram(path + "/tools/butterfly.exe");
    ordem2k.start();
    if (ordem2k.waitForFinished() && QFile::exists(path + "/tools/ORDEM2000.res")){
        ui.statusBar->showMessage(QString::fromLocal8Bit("完成"));
    }
    else{
        ui.statusBar->showMessage("butterfly Error...");
    }
    return;
}

void UIShell::applySettings()
{
    ui.statusBar->clearMessage();
#ifdef _DEBUG
    QString path = QString(HOME);
#else
    QString path = QCoreApplication::applicationDirPath();
#endif

    QFile cmd(path + "/tools/ORDEM2000.CMD");
    if (!cmd.open(QIODevice::WriteOnly | QIODevice::Text)){
        ui.statusBar->showMessage("CMD Error...");
        return;
    }

    //
    cmd.write("! file=ordem2000.cmd threshold=");
    cmd.write(ui.comboBox_Diameter->currentText().toLatin1());
    cmd.write(" generated by SCRA.\n");
    cmd.write("1");
    cmd.write(" ! type of assessment (0=telescope 1=spacecraft)\n");
    cmd.write(ui.comboBox_Year->currentText().toLatin1());
    cmd.write(" ! year of observation (1991-2030)\n");
    cmd.write(ui.checkBox_SMA_Ecc->isChecked() ? "1" : "2");
    cmd.write(" ! way to determine orbit (1=semi maj axis & eccentricity 2=apogee & perigee)\n");
    if (ui.checkBox_SMA_Ecc->isChecked()){
        cmd.write(ui.lineEdit_SMA_Apo->text().toLatin1());
        cmd.write(" ! semi-major axis (km)\n");
        cmd.write(ui.lineEdit_Ecc_Peri->text().toLatin1());
        cmd.write(" ! eccentricity (0 to 1.0)\n");
    }
    else{
        cmd.write(ui.lineEdit_Ecc_Peri->text().toLatin1());
        cmd.write(" ! perigee (km)\n");
        cmd.write(ui.lineEdit_SMA_Apo->text().toLatin1());
        cmd.write(" ! apogee (km)\n");
    }
    cmd.write(ui.checkBox_Argp->isChecked() ? "-1" : ui.lineEdit_Argp->text().toLatin1());
    cmd.write(" !argument of perigee(0 to 360deg, -1 = random)\n");
    cmd.write(ui.lineEdit_Inc->text().toLatin1());
    cmd.write(" ! inclination (0 to 180deg)\n");
    cmd.write(ui.lineEdit_Seg->text().toLatin1());
    cmd.write(" ! number of segments in orbit (integer>0)\n");
    cmd.write("TABLESC.DAT ! flux output file name (def=tablesc.dat)\n");
    cmd.write(path.replace("/","\\").toLatin1() + "\\tools\\results\\");
    cmd.write(" ! path to where to put results (def=ordem2000\\results\\)\n");
    cmd.write("1 ! Produce additional VREL files? (0=No 1=Yes)\n");
    cmd.close();

    //
    QStringList dats = QDir(path + "/tools/results/").entryList(QStringList()<<"*.dat");
    for (int i = 0; i < dats.size(); i++){
        QFile::remove(path + "/tools/results/" + dats.at(i));
    }
    return;
}

void UIShell::genSample()
{
    ui.statusBar->showMessage(QString::fromLocal8Bit("样本生成运行中..."));
#ifdef _DEBUG
    QString path = QString(HOME);
#else
    QString path = QCoreApplication::applicationDirPath();
#endif
    wchar_t *program = Py_DecodeLocale(QCoreApplication::arguments().at(0).toLatin1(), NULL);
    Py_SetProgramName(program);  /* optional but recommended */
    Py_Initialize();
    FILE* dotpy = _Py_fopen(QString(path).append("/tools/ordem2k.py").toLatin1(), "r");
    PyRun_SimpleFileEx(dotpy, "ordem2k.py", 1);
    Py_Finalize();
    PyMem_RawFree(program);
    
    if ( QFile::exists(QString(path).append("/output/sample.txt"))){
        QFile::remove(QString(path).append("/output/sample.txt"));
    }
    bool r = QFile::copy(QString(path).append("/tools/sample.txt"), QString(path).append("/output/sample.txt"));

    ui.statusBar->clearMessage();
    ui.statusBar->showMessage(QString(path).append("/output/sample.txt"));
    return;
}

void UIShell::riskAssessment()
{
    ui.statusBar->showMessage(QString::fromLocal8Bit("风险评估计算中..."));
#ifdef _DEBUG
    QString path = QString(HOME);
#else
    QString path = QCoreApplication::applicationDirPath();
#endif
    wchar_t *program = Py_DecodeLocale(QCoreApplication::arguments().at(0).toLatin1(), NULL);
    Py_SetProgramName(program);  /* optional but recommended */
    Py_Initialize();
    FILE* dotpy = _Py_fopen(QString(path).append("/tools/pnp.py").toLatin1(), "r");
    PyRun_SimpleFileEx(dotpy, "pnp.py", 1);
    Py_Finalize();
    PyMem_RawFree(program);

    if (QFile::exists(QString(path).append("/output/RiskAssessment.txt"))){
        QFile::remove(QString(path).append("/output/RiskAssessment.txt"));
    }
    bool r = QFile::copy(QString(path).append("/tools/RiskAssessment.txt"), QString(path).append("/output/RiskAssessment.txt"));

    ui.statusBar->clearMessage();
    ui.statusBar->showMessage(QString(path).append("/output/RiskAssessment.txt"));
    return;
}
