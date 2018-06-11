#ifndef UISHELL_H
#define UISHELL_H

#include <QtWidgets/QMainWindow>
#include "ui_uishell.h"

class UIShell : public QMainWindow
{
    Q_OBJECT

public:
    UIShell(QWidget *parent = 0);
    ~UIShell();

public slots:
    void sma_Ecc_fromto_Apo_Peri(int state);
    void sdModelTest();
    void applySettings();
    void genSample();
    void riskAssessment();

private:
    Ui::UIShellClass ui;
    QString qstring_SMA;
    QString qstring_Ecc;
    QString qstring_Apo;
    QString qstring_Peri;
};

#endif // UISHELL_H
