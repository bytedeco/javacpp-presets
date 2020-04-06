import java.io.File;

import org.bytedeco.javacpp.*;
import org.bytedeco.llvm.clang.*;
import org.bytedeco.llvm.global.clang;

/**
 * Reference
 * https://github.com/sabottenda/libclang-sample/blob/master/CodeComplete/CodeComplete.cc
 * 
 * @author Nico Hezel
 *
 */
public class ClangDiagnosisExample {

	public static void main(String[] args) throws Exception {

		// current clang version
		System.out.println(clang.clang_getClangVersion().getString());

		// Location of the *.cc file
		File file = new File("sample3.cc");
		System.out.println("Analyse file: "+ file); 

		// Command line parameters provided to clang.
		// https://clang.llvm.org/docs/ClangCommandLineReference.html
		String[] command_line_args = new String[] { };

		// Create an index containing all source files and essentially creating one executable
		// excludeDeclsFromPCH = 1, displayDiagnostics = 1
		CXIndex index = clang.clang_createIndex(1, 0);

		// Build a translation unit from the source file
		int num_unsaved_files = 0;
		CXUnsavedFile unsaved_files = new CXUnsavedFile();
		BytePointer source_filename = new BytePointer(file.toString());
		PointerPointer command_line_args_ptr = new PointerPointer(command_line_args);
		CXTranslationUnit translationUnit = new CXTranslationUnit();
		checkError(clang.clang_parseTranslationUnit2(index, source_filename, command_line_args_ptr, command_line_args.length,
				unsaved_files, num_unsaved_files, clang.CXTranslationUnit_None, translationUnit));

		// Analyze the source code and print some diagnostic informations.
		diagnosis(translationUnit);

		// dispose all allocated data
		clang.clang_disposeTranslationUnit(translationUnit);
		clang.clang_disposeIndex(index);
	}

	protected static void diagnosis(CXTranslationUnit translationUnit) {
		CXDiagnosticSet diagSet = clang.clang_getDiagnosticSetFromTU(translationUnit);
		int numDiag = clang.clang_getNumDiagnosticsInSet(diagSet);
		System.out.printf("numDiag: %d\n", numDiag);

		for (int i = 0; i < numDiag; i++) {
			CXDiagnostic diag = clang.clang_getDiagnosticInSet(diagSet, i);

			// show diagnosis spell
			System.out.printf("  Diagnosis: %s\n", clang.clang_getDiagnosticSpelling(diag).getString());

			show_diagnosis_format(diag);   // format
			show_diagnosis_severity(diag); // severity
			show_diagnosis_location(diag); // location
			show_diagnosis_category(diag); // category
			show_diagnosis_range(diag);    // range
			show_diagnosis_fixit(diag);    // fixit
			show_diagnosis_child(diag);    // child
			System.out.printf("\n");

			clang.clang_disposeDiagnostic(diag);
		}
		clang.clang_disposeDiagnosticSet(diagSet);
	}

	protected static void show_diagnosis_format(CXDiagnostic diag) {
		int formatOption =  clang.CXDiagnostic_DisplaySourceLocation |
							clang.CXDiagnostic_DisplayColumn |
							clang.CXDiagnostic_DisplaySourceRanges |
							clang.CXDiagnostic_DisplayOption |
							clang.CXDiagnostic_DisplayCategoryId|
							clang.CXDiagnostic_DisplayCategoryName;
		System.out.printf("  Format: %s\n", clang.clang_formatDiagnostic(diag, formatOption).getString());
	}

	protected static void show_diagnosis_severity(CXDiagnostic diag) {
		int severity = clang.clang_getDiagnosticSeverity(diag);
		String severityText;
		switch (severity) {
			case clang.CXDiagnostic_Ignored: severityText = "Ignored"; break;
			case clang.CXDiagnostic_Note:    severityText = "Note";    break;
			case clang.CXDiagnostic_Warning: severityText = "Warning"; break;
			case clang.CXDiagnostic_Error:   severityText = "Error";   break;
			case clang.CXDiagnostic_Fatal:   severityText = "Fatal";   break;
			default:                   		 severityText = "Unknown"; break;
		}
		System.out.printf("  Severity: %s\n", severityText);
	}

	protected static void show_diagnosis_location(CXDiagnostic diag) {
		CXSourceLocation loc = clang.clang_getDiagnosticLocation(diag);
		CXFile file = new CXFile();
		int[] line = new int[1], column = new int[1], offset = new int[1];
		clang.clang_getSpellingLocation(loc, file, line, column, offset);
		System.out.printf("  Location: %s:%d:%d:%d\n", 
				clang.clang_getFileName(file).getString(), line[0], column[0], offset[0]);
	}

	protected static void show_diagnosis_category(CXDiagnostic diag) {
		System.out.printf("  Category: %s\n", clang.clang_getDiagnosticCategoryText(diag).getString());
	}

	protected static void show_diagnosis_range(CXDiagnostic diag) {
		CXFile file = new CXFile();
		int[] line = new int[1], column = new int[1], offset = new int[1];

		int numRange = clang.clang_getDiagnosticNumRanges(diag);
		System.out.printf("  NumRange: %d\n", numRange);
		for (int j = 0; j < numRange; j++) {
			CXSourceRange range = clang.clang_getDiagnosticRange(diag, j);
			System.out.printf("    Range %d\n", j);

			CXSourceLocation start = clang.clang_getRangeStart(range);
			clang.clang_getSpellingLocation(start, file, line, column, offset);
			System.out.printf("      Start: %s:%d:%d:%d\n",
					clang.clang_getFileName(file).getString(), line[0], column[0], offset[0]);

			CXSourceLocation end = clang.clang_getRangeEnd(range);
			clang.clang_getSpellingLocation(end, file, line, column, offset);
			System.out.printf("      End: %s:%d:%d:%d\n",
					clang.clang_getFileName(file).getString(), line[0], column[0], offset[0]);
		}
	}

	protected static void show_diagnosis_fixit(CXDiagnostic diag) {
		int numFixit = clang.clang_getDiagnosticNumFixIts(diag);
		CXSourceRange range = new CXSourceRange();
		System.out.printf("  NumFixit: %d\n", numFixit);
		for (int j = 0; j < numFixit; j++) {
			CXString fixit = clang.clang_getDiagnosticFixIt(diag, j, range);
			System.out.printf("    Fixit: %s\n", fixit.getString());
		}
	}

	protected static void show_diagnosis_child(CXDiagnostic diag) {
		CXDiagnosticSet childDiagSet = clang.clang_getChildDiagnostics(diag);
		int numChildDiag = clang.clang_getNumDiagnosticsInSet(childDiagSet);
		System.out.printf("  NumChildDiag: %d\n", numChildDiag);

		// TODO: show child DiagnosticSet recursively(?)
	}

	/**
	 * Check for errors of the compilation process.
	 * 
	 * @param errorCode
	 * @throws Exception
	 */
	protected static void checkError(int errorCode) throws Exception {
		if(errorCode != clang.CXError_Success) {
			switch (errorCode) {
			case clang.CXError_InvalidArguments:
				throw new Exception("InvalidArguments");
			case clang.CXError_ASTReadError:
				throw new Exception("ASTReadError");
			case clang.CXError_Crashed:
				throw new Exception("Crashed");
			case clang.CXError_Failure:
				throw new Exception("Failure");
			}
		}
	}

}
