import java.io.File;

import org.bytedeco.javacpp.*;
import org.bytedeco.llvm.clang.*;
import org.bytedeco.llvm.global.clang;

/**
 * Reference
 * https://github.com/sabottenda/libclang-sample/blob/master/AST/ASTVisitor.cc
 * 
 * @author Nico Hezel
 *
 */
public class ClangASTVisitorExample {

	public static void main(String[] args) throws Exception {

		// current clang version
		System.out.println(clang.clang_getClangVersion().getString());

		// Location of the *.cc file
		File file = new File("sample1.cc");
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
				
		// traverse the elements
		CXClientData level = new CXClientData(new IntPointer(new int[] {0}));
		CXCursorVisitor visitor = new TemplateVisitor();
		CXCursor rootCursor  = clang.clang_getTranslationUnitCursor(translationUnit);
		clang.clang_visitChildren(rootCursor, visitor, level);

		// dispose all allocated data
		clang.clang_disposeTranslationUnit(translationUnit);
		clang.clang_disposeIndex(index);
	}

	/**
	 * Class with the callback method of the CXCursorVisitor.
	 */
	protected static class TemplateVisitor extends CXCursorVisitor {

		@Override
		public int call(CXCursor cursor, CXCursor parent, CXClientData client_data) {

			CXSourceLocation location = clang.clang_getCursorLocation(cursor);
			if(clang.clang_Location_isFromMainFile(location) == 0)
				return clang.CXChildVisit_Continue;

			int level = client_data.asByteBuffer().getInt();
			System.out.printf("  Level: %d\n", level);

			show_spell(cursor);
			show_linkage(cursor);
			show_cursor_kind(cursor);
			show_type(cursor);
			show_parent(cursor, parent);
			show_location(cursor);
			show_usr(cursor);
			show_included_file(cursor);
			System.out.println();

			CXClientData nextLevel = new CXClientData(new IntPointer(new int[] { level + 1 }));
			clang.clang_visitChildren(cursor, this, nextLevel); 

			return clang.CXChildVisit_Continue;
		}


		protected void show_spell(CXCursor cursor) {
			System.out.printf("  Text: %s\n", clang.clang_getCursorSpelling(cursor).getString());
		}

		protected void show_type(CXCursor cursor) {
			CXType type = clang.clang_getCursorType(cursor);
			System.out.printf("  Type: %s\n", clang.clang_getTypeSpelling(type).getString());
			System.out.printf("  TypeKind: %s\n", clang.clang_getTypeKindSpelling(type.kind()).getString());
		}

		protected void show_linkage(CXCursor cursor) {
			int linkage = clang.clang_getCursorLinkage(cursor);
			String linkageName;
			switch (linkage) {
				case clang.CXLinkage_Invalid:        linkageName = "Invalid"; break;
				case clang.CXLinkage_NoLinkage:      linkageName = "NoLinkage"; break;
				case clang.CXLinkage_Internal:       linkageName = "Internal"; break;
				case clang.CXLinkage_UniqueExternal: linkageName = "UniqueExternal"; break;
				case clang.CXLinkage_External:       linkageName = "External"; break;
				default:                       		 linkageName = "Unknown"; break;
			}
			System.out.printf("  Linkage: %s\n", linkageName);
		}

		protected void show_parent(CXCursor cursor, CXCursor parent) {
			CXCursor semaParent = clang.clang_getCursorSemanticParent(cursor);
			CXCursor lexParent  = clang.clang_getCursorLexicalParent(cursor);
			System.out.printf("  Parent: parent:%s semantic:%s lexicial:%s\n",
					clang.clang_getCursorSpelling(parent).getString(),
					clang.clang_getCursorSpelling(semaParent).getString(),
					clang.clang_getCursorSpelling(lexParent).getString());
		}
		
		protected void show_location(CXCursor cursor) {
			CXSourceLocation loc = clang.clang_getCursorLocation(cursor);
			CXFile file = new CXFile();			
			int[] line = new int[1], column = new int[1], offset = new int[1];
			clang.clang_getSpellingLocation(loc, file, line, column, offset);
			
			System.out.printf("  Location: %s:%d:%d:%d\n", 
				clang.clang_getFileName(file).getString(), line[0], column[0], offset[0]);		
		}

		protected void show_usr(CXCursor cursor) {
			System.out.printf("  USR: %s\n", clang.clang_getCursorUSR(cursor).getString());
		}

		protected void show_cursor_kind(CXCursor cursor) {
			int curKind  = clang.clang_getCursorKind(cursor);

			String type;
			if (clang.clang_isAttribute(curKind) == 1)            type = "Attribute";
			else if (clang.clang_isDeclaration(curKind) == 1)     type = "Declaration";
			else if (clang.clang_isExpression(curKind) == 1)      type = "Expression";
			else if (clang.clang_isInvalid(curKind) == 1)         type = "Invalid";
			else if (clang.clang_isPreprocessing(curKind) == 1)   type = "Preprocessing";
			else if (clang.clang_isReference(curKind) == 1)       type = "Reference";
			else if (clang.clang_isStatement(curKind) == 1)       type = "Statement";
			else if (clang.clang_isTranslationUnit(curKind) == 1) type = "TranslationUnit";
			else if (clang.clang_isUnexposed(curKind) == 1)       type = "Unexposed";
			else                               					  type = "Unknown";

			System.out.printf("  CursorKind: %s\n",  clang.clang_getCursorKindSpelling(curKind).getString());
			System.out.printf("  CursorKindType: %s\n", type);
		}

		protected void show_included_file(CXCursor cursor) {
			CXFile included = clang.clang_getIncludedFile(cursor);
			if (included == null) return;
			System.out.printf(" included file: %s\n", clang.clang_getFileName(included).getString());
		}

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
