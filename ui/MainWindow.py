# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 764)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/icons/isat_bg_50x25.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setEnabled(True)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 25))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.menubar.setFont(font)
        self.menubar.setAutoFillBackground(False)
        self.menubar.setDefaultUp(False)
        self.menubar.setNativeMenuBar(True)
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.menuFile.setFont(font)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.menuView.setFont(font)
        self.menuView.setObjectName("menuView")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.menuAbout.setFont(font)
        self.menuAbout.setObjectName("menuAbout")
        self.menuLaguage = QtWidgets.QMenu(self.menuAbout)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icon/icons/翻译_translate.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menuLaguage.setIcon(icon1)
        self.menuLaguage.setObjectName("menuLaguage")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.menuTools.setFont(font)
        self.menuTools.setObjectName("menuTools")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.menuEdit.setFont(font)
        self.menuEdit.setObjectName("menuEdit")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.toolBar.setFont(font)
        self.toolBar.setIconSize(QtCore.QSize(24, 24))
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.toolBar.setFloatable(False)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.info_dock = QtWidgets.QDockWidget(MainWindow)
        self.info_dock.setMinimumSize(QtCore.QSize(85, 43))
        self.info_dock.setFeatures(QtWidgets.QDockWidget.AllDockWidgetFeatures)
        self.info_dock.setObjectName("info_dock")
        self.dockWidgetContents_2 = QtWidgets.QWidget()
        self.dockWidgetContents_2.setObjectName("dockWidgetContents_2")
        self.info_dock.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.info_dock)
        self.labels_dock = QtWidgets.QDockWidget(MainWindow)
        self.labels_dock.setMinimumSize(QtCore.QSize(85, 43))
        self.labels_dock.setFeatures(QtWidgets.QDockWidget.AllDockWidgetFeatures)
        self.labels_dock.setObjectName("labels_dock")
        self.dockWidgetContents_3 = QtWidgets.QWidget()
        self.dockWidgetContents_3.setObjectName("dockWidgetContents_3")
        self.labels_dock.setWidget(self.dockWidgetContents_3)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.labels_dock)
        self.files_dock = QtWidgets.QDockWidget(MainWindow)
        self.files_dock.setObjectName("files_dock")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.files_dock.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.files_dock)
        self.actionOpen_dir = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icon/icons/照片_pic.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen_dir.setIcon(icon2)
        self.actionOpen_dir.setObjectName("actionOpen_dir")
        self.actionZoom_in = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icon/icons/放大_zoom-in.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionZoom_in.setIcon(icon3)
        self.actionZoom_in.setObjectName("actionZoom_in")
        self.actionZoom_out = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icon/icons/缩小_zoom-out.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionZoom_out.setIcon(icon4)
        self.actionZoom_out.setObjectName("actionZoom_out")
        self.actionFit_wiondow = QtWidgets.QAction(MainWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icon/icons/全宽_fullwidth.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFit_wiondow.setIcon(icon5)
        self.actionFit_wiondow.setObjectName("actionFit_wiondow")
        self.actionSetting = QtWidgets.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icon/icons/设置_setting-two.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSetting.setIcon(icon6)
        self.actionSetting.setObjectName("actionSetting")
        self.actionExit = QtWidgets.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/icon/icons/开关_power.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionExit.setIcon(icon7)
        self.actionExit.setObjectName("actionExit")
        self.actionSave_dir = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/icon/icons/文件夹-开_folder-open.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave_dir.setIcon(icon8)
        self.actionSave_dir.setObjectName("actionSave_dir")
        self.actionSave = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/icon/icons/保存_save.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave.setIcon(icon9)
        self.actionSave.setObjectName("actionSave")
        self.actionPrev = QtWidgets.QAction(MainWindow)
        self.actionPrev.setCheckable(False)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/icon/icons/上一步_back.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPrev.setIcon(icon10)
        self.actionPrev.setMenuRole(QtWidgets.QAction.TextHeuristicRole)
        self.actionPrev.setPriority(QtWidgets.QAction.NormalPriority)
        self.actionPrev.setObjectName("actionPrev")
        self.actionNext = QtWidgets.QAction(MainWindow)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(":/icon/icons/下一步_next.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNext.setIcon(icon11)
        self.actionNext.setObjectName("actionNext")
        self.actionShortcut = QtWidgets.QAction(MainWindow)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(":/icon/icons/键盘_keyboard-one.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionShortcut.setIcon(icon12)
        self.actionShortcut.setObjectName("actionShortcut")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap(":/icon/icons/我的_me.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAbout.setIcon(icon13)
        self.actionAbout.setObjectName("actionAbout")
        self.actionSegment_anything = QtWidgets.QAction(MainWindow)
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap(":/icon/icons/M_Favicon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSegment_anything.setIcon(icon14)
        self.actionSegment_anything.setObjectName("actionSegment_anything")
        self.actionDelete = QtWidgets.QAction(MainWindow)
        self.actionDelete.setEnabled(False)
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap(":/icon/icons/删除_delete.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionDelete.setIcon(icon15)
        self.actionDelete.setObjectName("actionDelete")
        self.actionBit_map = QtWidgets.QAction(MainWindow)
        self.actionBit_map.setCheckable(False)
        self.actionBit_map.setIcon(icon2)
        self.actionBit_map.setObjectName("actionBit_map")
        self.actionEdit = QtWidgets.QAction(MainWindow)
        self.actionEdit.setEnabled(False)
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap(":/icon/icons/编辑_edit.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionEdit.setIcon(icon16)
        self.actionEdit.setObjectName("actionEdit")
        self.actionTo_top = QtWidgets.QAction(MainWindow)
        self.actionTo_top.setEnabled(False)
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap(":/icon/icons/去顶部_to-top.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTo_top.setIcon(icon17)
        self.actionTo_top.setObjectName("actionTo_top")
        self.actionTo_bottom = QtWidgets.QAction(MainWindow)
        self.actionTo_bottom.setEnabled(False)
        icon18 = QtGui.QIcon()
        icon18.addPixmap(QtGui.QPixmap(":/icon/icons/去底部_to-bottom.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTo_bottom.setIcon(icon18)
        self.actionTo_bottom.setObjectName("actionTo_bottom")
        self.actionToVOC = QtWidgets.QAction(MainWindow)
        icon19 = QtGui.QIcon()
        icon19.addPixmap(QtGui.QPixmap(":/icon/icons/VOC_32x32.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionToVOC.setIcon(icon19)
        self.actionToVOC.setWhatsThis("")
        self.actionToVOC.setObjectName("actionToVOC")
        self.actionChinese = QtWidgets.QAction(MainWindow)
        self.actionChinese.setCheckable(True)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.actionChinese.setFont(font)
        self.actionChinese.setObjectName("actionChinese")
        self.actionEnglish = QtWidgets.QAction(MainWindow)
        self.actionEnglish.setCheckable(True)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.actionEnglish.setFont(font)
        self.actionEnglish.setObjectName("actionEnglish")
        self.actionBackspace = QtWidgets.QAction(MainWindow)
        icon20 = QtGui.QIcon()
        icon20.addPixmap(QtGui.QPixmap(":/icon/icons/删除_delete-two.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionBackspace.setIcon(icon20)
        self.actionBackspace.setObjectName("actionBackspace")
        self.actionCancel = QtWidgets.QAction(MainWindow)
        icon21 = QtGui.QIcon()
        icon21.addPixmap(QtGui.QPixmap(":/icon/icons/关闭_close-one.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionCancel.setIcon(icon21)
        self.actionCancel.setObjectName("actionCancel")
        self.actionFinish = QtWidgets.QAction(MainWindow)
        icon22 = QtGui.QIcon()
        icon22.addPixmap(QtGui.QPixmap(":/icon/icons/校验_check-one.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFinish.setIcon(icon22)
        self.actionFinish.setObjectName("actionFinish")
        self.actionPolygon = QtWidgets.QAction(MainWindow)
        icon23 = QtGui.QIcon()
        icon23.addPixmap(QtGui.QPixmap(":/icon/icons/锚点_anchor.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPolygon.setIcon(icon23)
        self.actionPolygon.setObjectName("actionPolygon")
        self.actionVisible = QtWidgets.QAction(MainWindow)
        icon24 = QtGui.QIcon()
        icon24.addPixmap(QtGui.QPixmap(":/icon/icons/眼睛_eyes.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionVisible.setIcon(icon24)
        self.actionVisible.setObjectName("actionVisible")
        self.actionToCOCO = QtWidgets.QAction(MainWindow)
        icon25 = QtGui.QIcon()
        icon25.addPixmap(QtGui.QPixmap(":/icon/icons/coco.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionToCOCO.setIcon(icon25)
        self.actionToCOCO.setObjectName("actionToCOCO")
        self.actionFromCOCO = QtWidgets.QAction(MainWindow)
        self.actionFromCOCO.setIcon(icon25)
        self.actionFromCOCO.setObjectName("actionFromCOCO")
        self.actionTo_LabelMe = QtWidgets.QAction(MainWindow)
        icon26 = QtGui.QIcon()
        icon26.addPixmap(QtGui.QPixmap(":/icon/icons/labelme_32x32.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTo_LabelMe.setIcon(icon26)
        self.actionTo_LabelMe.setObjectName("actionTo_LabelMe")
        self.actionCLIPSEG = QtWidgets.QAction(MainWindow)
        self.actionCLIPSEG.setObjectName("actionCLIPSEG")
        self.actionRVSA = QtWidgets.QAction(MainWindow)
        self.actionRVSA.setObjectName("actionRVSA")
        self.actionSAMS = QtWidgets.QAction(MainWindow)
        self.actionSAMS.setObjectName("actionSAMS")
        self.menuFile.addAction(self.actionOpen_dir)
        self.menuFile.addAction(self.actionSave_dir)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionPrev)
        self.menuFile.addAction(self.actionNext)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSetting)
        self.menuFile.addAction(self.actionExit)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionZoom_in)
        self.menuView.addAction(self.actionZoom_out)
        self.menuView.addAction(self.actionFit_wiondow)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionBit_map)
        self.menuView.addSeparator()
        self.menuLaguage.addAction(self.actionChinese)
        self.menuLaguage.addAction(self.actionEnglish)
        self.menuAbout.addAction(self.menuLaguage.menuAction())
        self.menuAbout.addAction(self.actionShortcut)
        self.menuAbout.addAction(self.actionAbout)
        self.menuTools.addAction(self.actionToVOC)
        self.menuTools.addAction(self.actionToCOCO)
        self.menuTools.addAction(self.actionTo_LabelMe)
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.actionFromCOCO)
        self.menuEdit.addAction(self.actionSegment_anything)
        self.menuEdit.addAction(self.actionPolygon)
        self.menuEdit.addAction(self.actionBackspace)
        self.menuEdit.addAction(self.actionFinish)
        self.menuEdit.addAction(self.actionCancel)
        self.menuEdit.addSeparator()
        self.menuEdit.addAction(self.actionTo_top)
        self.menuEdit.addAction(self.actionTo_bottom)
        self.menuEdit.addSeparator()
        self.menuEdit.addAction(self.actionEdit)
        self.menuEdit.addAction(self.actionDelete)
        self.menuEdit.addAction(self.actionSave)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())
        self.toolBar.addAction(self.actionPrev)
        self.toolBar.addAction(self.actionNext)
        self.toolBar.addSeparator()
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSegment_anything)
        self.toolBar.addAction(self.actionPolygon)
        self.toolBar.addAction(self.actionBackspace)
        self.toolBar.addAction(self.actionFinish)
        self.toolBar.addAction(self.actionCancel)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionTo_top)
        self.toolBar.addAction(self.actionTo_bottom)
        self.toolBar.addAction(self.actionEdit)
        self.toolBar.addAction(self.actionDelete)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionZoom_in)
        self.toolBar.addAction(self.actionZoom_out)
        self.toolBar.addAction(self.actionFit_wiondow)
        self.toolBar.addAction(self.actionBit_map)
        self.toolBar.addAction(self.actionVisible)
        self.toolBar.addAction(self.actionCLIPSEG)
        self.toolBar.addAction(self.actionRVSA)
        self.toolBar.addAction(self.actionSAMS)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ISAT"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuAbout.setTitle(_translate("MainWindow", "Help"))
        self.menuLaguage.setTitle(_translate("MainWindow", "Laguage"))
        self.menuTools.setTitle(_translate("MainWindow", "Tools"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.info_dock.setWindowTitle(_translate("MainWindow", "Info"))
        self.labels_dock.setWindowTitle(_translate("MainWindow", "Labels"))
        self.files_dock.setWindowTitle(_translate("MainWindow", "Files"))
        self.actionOpen_dir.setText(_translate("MainWindow", "Images dir"))
        self.actionOpen_dir.setStatusTip(_translate("MainWindow", "Open images dir."))
        self.actionZoom_in.setText(_translate("MainWindow", "Zoom in"))
        self.actionZoom_in.setStatusTip(_translate("MainWindow", "Zoom in."))
        self.actionZoom_out.setText(_translate("MainWindow", "Zoom out"))
        self.actionZoom_out.setStatusTip(_translate("MainWindow", "Zoom out."))
        self.actionFit_wiondow.setText(_translate("MainWindow", "Fit window"))
        self.actionFit_wiondow.setToolTip(_translate("MainWindow", "Fit window"))
        self.actionFit_wiondow.setStatusTip(_translate("MainWindow", "Fit window."))
        self.actionFit_wiondow.setShortcut(_translate("MainWindow", "F"))
        self.actionSetting.setText(_translate("MainWindow", "Setting"))
        self.actionSetting.setStatusTip(_translate("MainWindow", "Setting."))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setToolTip(_translate("MainWindow", "Exit"))
        self.actionExit.setStatusTip(_translate("MainWindow", "Exit."))
        self.actionSave_dir.setText(_translate("MainWindow", "Label dir"))
        self.actionSave_dir.setStatusTip(_translate("MainWindow", "Open label dir."))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave.setStatusTip(_translate("MainWindow", "Save annotation."))
        self.actionSave.setShortcut(_translate("MainWindow", "S"))
        self.actionPrev.setText(_translate("MainWindow", "Prev image"))
        self.actionPrev.setToolTip(_translate("MainWindow", "Prev image"))
        self.actionPrev.setStatusTip(_translate("MainWindow", "Prev image."))
        self.actionPrev.setShortcut(_translate("MainWindow", "A"))
        self.actionNext.setText(_translate("MainWindow", "Next image"))
        self.actionNext.setToolTip(_translate("MainWindow", "Next image"))
        self.actionNext.setStatusTip(_translate("MainWindow", "Next image."))
        self.actionNext.setShortcut(_translate("MainWindow", "D"))
        self.actionShortcut.setText(_translate("MainWindow", "Shortcut"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionSegment_anything.setText(_translate("MainWindow", "Segment anything"))
        self.actionSegment_anything.setToolTip(_translate("MainWindow", "Segment anything"))
        self.actionSegment_anything.setStatusTip(_translate("MainWindow", "Quick annotate using Segment anything."))
        self.actionSegment_anything.setShortcut(_translate("MainWindow", "Q"))
        self.actionDelete.setText(_translate("MainWindow", "Delete"))
        self.actionDelete.setToolTip(_translate("MainWindow", "Delete polygon"))
        self.actionDelete.setStatusTip(_translate("MainWindow", "Delete polygon."))
        self.actionDelete.setShortcut(_translate("MainWindow", "Del"))
        self.actionBit_map.setText(_translate("MainWindow", "Bit map"))
        self.actionBit_map.setStatusTip(_translate("MainWindow", "Show instance or segmeent state."))
        self.actionBit_map.setShortcut(_translate("MainWindow", "Space"))
        self.actionEdit.setText(_translate("MainWindow", "Edit"))
        self.actionEdit.setToolTip(_translate("MainWindow", "Edit polygon"))
        self.actionEdit.setStatusTip(_translate("MainWindow", "Edit polygon attribute."))
        self.actionTo_top.setText(_translate("MainWindow", "To top"))
        self.actionTo_top.setToolTip(_translate("MainWindow", "Move polygon to top layer"))
        self.actionTo_top.setStatusTip(_translate("MainWindow", "Move polygon to top layer."))
        self.actionTo_top.setShortcut(_translate("MainWindow", "T"))
        self.actionTo_bottom.setText(_translate("MainWindow", "To bottom"))
        self.actionTo_bottom.setToolTip(_translate("MainWindow", "Move polygon to bottom layer"))
        self.actionTo_bottom.setStatusTip(_translate("MainWindow", "Move polygon to bottom layer."))
        self.actionTo_bottom.setShortcut(_translate("MainWindow", "B"))
        self.actionToVOC.setText(_translate("MainWindow", "To VOC"))
        self.actionToVOC.setToolTip(_translate("MainWindow", "Convert ISAT to VOC"))
        self.actionToVOC.setStatusTip(_translate("MainWindow", "Convert ISAT jsons to VOC png images."))
        self.actionChinese.setText(_translate("MainWindow", "中文"))
        self.actionEnglish.setText(_translate("MainWindow", "English"))
        self.actionBackspace.setText(_translate("MainWindow", "Backspace"))
        self.actionBackspace.setToolTip(_translate("MainWindow", "Backspace"))
        self.actionBackspace.setStatusTip(_translate("MainWindow", "Backspace."))
        self.actionBackspace.setShortcut(_translate("MainWindow", "Z"))
        self.actionCancel.setText(_translate("MainWindow", "Cancel"))
        self.actionCancel.setToolTip(_translate("MainWindow", "Annotate canceled"))
        self.actionCancel.setStatusTip(_translate("MainWindow", "Annotate canceled."))
        self.actionCancel.setShortcut(_translate("MainWindow", "Esc"))
        self.actionFinish.setText(_translate("MainWindow", "Finish"))
        self.actionFinish.setToolTip(_translate("MainWindow", "Annotate finished"))
        self.actionFinish.setStatusTip(_translate("MainWindow", "Annotate finished."))
        self.actionFinish.setShortcut(_translate("MainWindow", "E"))
        self.actionPolygon.setText(_translate("MainWindow", "Polygon"))
        self.actionPolygon.setToolTip(_translate("MainWindow", "Draw polygon"))
        self.actionPolygon.setStatusTip(_translate("MainWindow", "Accurately annotate by drawing polygon. "))
        self.actionPolygon.setShortcut(_translate("MainWindow", "C"))
        self.actionVisible.setText(_translate("MainWindow", "Visible"))
        self.actionVisible.setStatusTip(_translate("MainWindow", "Visible"))
        self.actionVisible.setShortcut(_translate("MainWindow", "V"))
        self.actionToCOCO.setText(_translate("MainWindow", "To COCO"))
        self.actionToCOCO.setToolTip(_translate("MainWindow", "Convert ISAT to COCO"))
        self.actionToCOCO.setStatusTip(_translate("MainWindow", "Convert ISAT jsons to COCO json."))
        self.actionFromCOCO.setText(_translate("MainWindow", "From COCO"))
        self.actionFromCOCO.setToolTip(_translate("MainWindow", "Convert COCO to ISAT"))
        self.actionFromCOCO.setStatusTip(_translate("MainWindow", "Convert COCO json to ISAT jsons."))
        self.actionTo_LabelMe.setText(_translate("MainWindow", "To LabelMe"))
        self.actionTo_LabelMe.setToolTip(_translate("MainWindow", "Convert ISAT to LabelMe"))
        self.actionTo_LabelMe.setStatusTip(_translate("MainWindow", "Convert ISAT jsons to LabelMe jsons."))
        self.actionCLIPSEG.setText(_translate("MainWindow", "CLIPSEG"))
        self.actionCLIPSEG.setToolTip(_translate("MainWindow", "CLIPSEG"))
        self.actionRVSA.setText(_translate("MainWindow", "RVSA"))
        self.actionSAMS.setText(_translate("MainWindow", "SAMS"))
import icons_rc
