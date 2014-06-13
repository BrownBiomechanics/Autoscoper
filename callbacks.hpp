// ----------------------------------
// Copyright (c) 2011, Brown University
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// (1) Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// 
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
// 
// (3) Neither the name of Brown University nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY BROWN UNIVERSITY “AS IS” WITH NO
// WARRANTIES OR REPRESENTATIONS OF ANY KIND WHATSOEVER EITHER EXPRESS OR
// IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTY OF DESIGN OR
// MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EACH OF WHICH ARE
// SPECIFICALLY DISCLAIMED, NOR ANY WARRANTY OR REPRESENTATIONS THAT THE
// SOFTWARE IS ERROR FREE OR THAT THE SOFTWARE WILL NOT INFRINGE ANY
// PATENT, COPYRIGHT, TRADEMARK, OR OTHER THIRD PARTY PROPRIETARY RIGHTS.
// IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY OR CAUSE OF ACTION, WHETHER IN CONTRACT,
// STRICT LIABILITY, TORT, NEGLIGENCE OR OTHERWISE, ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE. ANY RECIPIENT OR USER OF THIS SOFTWARE ACKNOWLEDGES THE
// FOREGOING, AND ACCEPTS ALL RISKS AND LIABILITIES THAT MAY ARISE FROM
// THEIR USE OF THE SOFTWARE.
// ---------------------------------

#include <gtk/gtk.h>

void
on_toggle_filter(gpointer filter, bool toggled);

void
on_toggle_renderer(gpointer view, gint type, bool toggled);

void
on_remove_filter_activate(GtkWidget* menu_item, gpointer data);

void
on_xromm_drr_renderer_properties_dialog_sample_distance_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data);

void
on_xromm_drr_renderer_properties_dialog_intensity_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data);

void
on_xromm_drr_renderer_properties_dialog_cutoff_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data);

void
on_xromm_sobel_properties_dialog_scale_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data);

void
on_xromm_sobel_properties_dialog_blend_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data);

void
on_xromm_contrast_properties_dialog_alpha_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data);

void
on_xromm_contrast_properties_dialog_beta_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data);

void
on_xromm_gaussian_properties_dialog_radius_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data);

void
on_xromm_sharpen_properties_dialog_radius_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data);

void
on_xromm_sharpen_properties_dialog_contrast_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data);

void
on_new_filter_activate                  (GtkWidget*     menu_item,
                                         gpointer       data);



void
on_export_view_activate                 (GtkWidget*     menu_item,
                                         gpointer       data);



void
on_import_view_activate                 (GtkWidget*     menu_item,
                                         gpointer       data);

// Added by glade

gboolean
on_xromm_markerless_tracking_window_delete_event        
                                       (GtkWidget       *widget,
                                        GdkEvent        *event,
                                        gpointer         user_data);

void
on_new1_activate                       (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_open1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_save1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_save_as1_activate                   (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_quit1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_cut1_activate                       (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_copy1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_paste1_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_delete1_activate                    (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_about1_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_toolbar_new_button_clicked
                                       (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_toolbar_open_button_clicked
                                       (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_toolbar_save_button_clicked
                                       (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_saveas_button_clicked
                                       (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_toolbar_add_button_clicked
                                       (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_drawingarea1_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_drawingarea1_configure_event
                                       (GtkWidget       *widget,
                                        GdkEventConfigure *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_drawingarea1_expose_event
                                       (GtkWidget       *widget,
                                        GdkEventExpose  *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_drawingarea1_button_press_event
                                       (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_drawingarea1_button_release_event
                                       (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_drawingarea1_motion_notify_event
                                       (GtkWidget       *widget,
                                        GdkEventMotion  *event,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_drawingarea2_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_drawingarea2_configure_event
                                       (GtkWidget       *widget,
                                        GdkEventConfigure *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_drawingarea2_expose_event
                                       (GtkWidget       *widget,
                                        GdkEventExpose  *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_drawingarea2_button_press_event
                                       (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_drawingarea2_button_release_event
                                       (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_drawingarea2_motion_notify_event
                                       (GtkWidget       *widget,
                                        GdkEventMotion  *event,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_drawingarea_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_graph_drawingarea_configure_event
                                       (GtkWidget       *widget,
                                        GdkEventConfigure *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_graph_drawingarea_expose_event
                                       (GtkWidget       *widget,
                                        GdkEventExpose  *event,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_drawingarea1_toolbar_combo_box_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_drawingarea1_toolbar_combo_box_changed
                                       (GtkComboBox     *combobox,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_drawingarea2_toolbar_combo_box_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_drawingarea2_toolbar_combo_box_changed
                                       (GtkComboBox     *combobox,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_drawingarea1_scroll_event
                                       (GtkWidget       *widget,
                                        GdkEvent        *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_drawingarea2_scroll_event
                                       (GtkWidget       *widget,
                                        GdkEvent        *event,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_toolbar_translate_radiobutton_toggled
                                       (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_toolbar_rotate_radiobutton_toggled
                                       (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_drawingarea1_toolbar_global_radiobutton_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_drawingarea1_toolbar_inplane_radiobutton_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_drawingarea2_toolbar_global_radiobutton_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_drawingarea2_toolbar_inplane_radiobutton_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_show_grid_activate                  (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_toggletoolbutton1_toggled           (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_timeline_value_changed
                                       (GtkRange        *range,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_timeline_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_x_entry_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_x_entry_editing_done
                                       (GtkCellEditable *celleditable,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_x_entry_changed
                                       (GtkEditable     *editable,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_y_entry_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_z_entry_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_yaw_entry_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_pitch_entry_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_roll_entry_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_x_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_x_check_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_x_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_y_check_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_y_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_z_check_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_z_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_yaw_check_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_yaw_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_pitch_check_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_pitch_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_roll_check_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_roll_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_x_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_y_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_z_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_yaw_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_pitch_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_roll_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_roll_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_timeline_min_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_timeline_min_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_timeline_max_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_timeline_max_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_notebook_volumes_treeview_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_notebook_views_treeview_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_notebook_volumes_treeview_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_toolbar_track_button_clicked
                                        (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_tracking_from_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_tracking_to_spinbutton_value_changed
                                        (GtkSpinButton   *spinbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_tracking_from_spinbutton_realize
                                        (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_tracking_to_spinbutton_realize
                                        (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_tracking_sobel_sensitivity_scale_value_changed
                                        (GtkRange        *range,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_tracking_sobel_blend_scale_value_changed
                                        (GtkRange        *range,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_timeline_prev_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_timeline_stop_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_timeline_play_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_timeline_next_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_notebook_trial_treeview_realize
                                        (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_notebook_realize
                                        (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_window_realize
                                        (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_current_radiobutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_tracking_previous_radiobutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_tracking_extrap_radiobutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_graph_toolbar_realize
                                        (GtkWidget       *widget,
                                        gpointer         user_data);

void
on_cancelbutton1_clicked               (GtkButton       *button,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_menubar_activate_current
                                        (GtkMenuShell    *menushell,
                                        gboolean         force_hide,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_menubar_selection_done
                                        (GtkMenuShell    *menushell,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_toolbar_openbutton_clicked
                                        (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_new_trial1_activate                  (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_open_trial1_activate                 (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_save_trial1_activate                 (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_save_as_trial1_activate              (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_save_tracking1_activate              (GtkMenuItem     *menuitem,
                                        gpointer         user_data);


void
on_load_tracking1_activate              (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_toolbar_openbutton_clicked
                                        (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

void
on_xromm_markerless_toolbar_retrack_button_clicked
                                        (GtkToolButton   *toolbutton,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_window_key_press_event
                                        (GtkWidget       *widget,
                                        GdkEventKey     *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_window_key_release_event
                                        (GtkWidget       *widget,
                                        GdkEventKey     *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_window_key_press_event
                                        (GtkWidget       *widget,
                                        GdkEventKey     *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_window_key_release_event
                                        (GtkWidget       *widget,
                                        GdkEventKey     *event,
                                        gpointer         user_data);

void
on_tracking1_activate                  (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_Import_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_export_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_copy_activate                       (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_reset_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_import_tracking_activate            (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_export_tracking_activate            (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_copy_tracking_activate              (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_reset_tracking_activate             (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_import_tracking_activate            (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_export_tracking_activate            (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_reset_tracking_activate             (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_lock_frames1_activate               (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_unlock_frames1_activate             (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_copy_frames1_activate               (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_graph_drawingarea_button_press_event
                                        (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_graph_drawingarea_button_release_event
                                        (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data);

gboolean
on_xromm_markerless_tracking_graph_drawingarea_motion_notify_event
                                        (GtkWidget       *widget,
                                        GdkEventMotion  *event,
                                        gpointer         user_data);

void
on_cut1_activate                       (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_paste1_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_delete1_activate                    (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_xromm_markerless_tracking_spline_radiobutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data);

void
on_insert_key1_activate                (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_break_tangents1_activate            (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_undo1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_redo1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data);

void
on_smooth_tangents1_activate           (GtkMenuItem     *menuitem,
                                        gpointer         user_data);
