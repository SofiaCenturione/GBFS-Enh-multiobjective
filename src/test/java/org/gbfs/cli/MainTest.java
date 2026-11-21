package org.gbfs.cli;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MainTest {
    @Test
    void allFeaturesRunWritesPythonContractArtifacts() throws Exception {
        Path outputRoot = Path.of("results");
        Path runDir = outputRoot.resolve("dataset_01").resolve("all").resolve("run_01");

        String[] args = {
                "--dataset_idx", "1",
                "--run", "1",
                "--algo", "all",
                "--out_root", outputRoot.toString(),
                "--data_dir", "Data",
                "--cv_folds", "3",
                "--knn_eval_k", "3"
        };
        assertEquals(0, Main.run(args));

        assertTrue(Files.isRegularFile(runDir.resolve("run_info.json")));
        assertTrue(Files.isRegularFile(runDir.resolve("run_meta.json")));
        assertTrue(Files.isRegularFile(runDir.resolve("front_train_cv.csv")));
        assertTrue(Files.isRegularFile(runDir.resolve("test_points.csv")));
        assertTrue(Files.isRegularFile(runDir.resolve("selected_features.txt")));
        assertTrue(Files.readString(runDir.resolve("front_train_cv.csv"))
                .startsWith("dataset_idx,dataset,method,run,point_id,fnum,fRatio,acc_train_cv,eRate_train_cv,red,fset"));
        assertTrue(Files.readString(runDir.resolve("test_points.csv"))
                .contains("KNN_ALL"));

        Files.writeString(runDir.resolve("error.txt"), "stale failure");
        Files.writeString(runDir.resolve("unmanaged.txt"), "keep this file");
        Files.writeString(runDir.resolve("selected_features.txt"), "stale features");
        assertEquals(0, Main.run(args));
        assertTrue(!Files.exists(runDir.resolve("error.txt")));
        assertEquals("keep this file", Files.readString(runDir.resolve("unmanaged.txt")));
        assertEquals("0 1 2 3 4 5 6 7 8", Files.readString(runDir.resolve("selected_features.txt")));
        Files.delete(runDir.resolve("unmanaged.txt"));
    }

    @Test
    void enhancedRunWritesNormalAndBuddySelections() throws Exception {
        String algo = "enh:probabilistic:rc_greedy";
        assertEquals(0, Main.run(new String[] {
                "--dataset_idx", "1",
                "--run", "1",
                "--algo", algo,
                "--out_root", "results",
                "--data_dir", "Data",
                "--cv_folds", "3",
                "--knn_eval_k", "3"
        }));

        Path runDir = Path.of("results").resolve("dataset_01")
                .resolve("enh__probabilistic__rc_greedy__beta_2__tau_0p3__lam_0p5")
                .resolve("run_01");
        String testPoints = Files.readString(runDir.resolve("test_points.csv"));
        assertTrue(testPoints.contains(",normal,"));
        assertTrue(testPoints.contains(",buddy,"));
        assertTrue(testPoints.startsWith(
                "dataset_idx,dataset,method,run,test_mode,acc_test,eRate_test,fnum,fRatio,red,time_total,fset,"
                        + "init_mode,ks_mode,post_mode,prob_beta,ks_rc_tau,buddy_lambda"));
        assertTrue(Files.readString(runDir.resolve("front_train_cv.csv"))
                .startsWith("dataset_idx,dataset,method,run,point_id,fnum,fRatio,acc_train_cv,eRate_train_cv,red,fset,"
                        + "init_mode,ks_mode,prob_beta,ks_rc_tau,buddy_lambda"));
        Path comboDir = runDir.resolve("gbfs_enhanced_raw").resolve("01").resolve("run_01")
                .resolve("probabilistic").resolve("ks_rc_greedy");
        assertTrue(Files.isRegularFile(comboDir.resolve("S0_selected.txt")));
        assertTrue(Files.isRegularFile(comboDir.resolve("pareto_masks.npy")));
        assertTrue(Files.isRegularFile(comboDir.resolve("pareto_objs.npy")));
        assertTrue(Files.isRegularFile(comboDir.resolve("kNeigh_chosen.npy")));
        assertTrue(Files.isRegularFile(comboDir.resolve("solver_meta.json")));
        assertTrue(Files.isRegularFile(comboDir.resolve("post_normal").resolve("selected_features.txt")));
        assertTrue(Files.isRegularFile(comboDir.resolve("post_normal").resolve("post_metrics.json")));
        assertTrue(Files.isRegularFile(comboDir.resolve("post_buddy").resolve("selected_features.txt")));
        assertTrue(Files.isRegularFile(comboDir.resolve("post_buddy").resolve("post_metrics.json")));
        assertTrue(Files.isRegularFile(runDir.resolve("gbfs_enhanced_raw").resolve("01")
                .resolve("ablation_summary.csv")));
        assertTrue(!Files.exists(runDir.resolve("selected_features_post_normal.txt")));
        assertTrue(!Files.exists(runDir.resolve("post_normal")));
        String metadata = Files.readString(runDir.resolve("run_meta.json"));
        assertTrue(metadata.contains("\"cfg\": {"));
        assertTrue(metadata.contains("\"pop\": 20"));
        assertTrue(metadata.contains("\"paths\": {"));
        assertTrue(metadata.contains("\"combo_dir\":"));
        assertTrue(Files.readAllLines(runDir.resolve("front_train_cv.csv")).size() > 1);
    }
}
