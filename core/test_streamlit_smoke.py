import compileall
from pathlib import Path
from unittest import TestCase


class StreamlitSmokeTests(TestCase):
	def setUp(self):
		self.project_root = Path(__file__).resolve().parent.parent
		self.streamlit_root = self.project_root / 'streamlit_app'
		self.page_files = [
			self.streamlit_root / 'pages' / 'overview.py',
			self.streamlit_root / 'pages' / 'realtime_monitor.py',
			self.streamlit_root / 'pages' / 'model_metrics.py',
			self.streamlit_root / 'pages' / 'review_queue.py',
			self.streamlit_root / 'pages' / 'truth_predict_analysis.py',
		]

	def test_streamlit_sources_compile(self):
		compiled = compileall.compile_dir(str(self.streamlit_root), force=True, quiet=1)
		self.assertTrue(compiled, 'Streamlit modules must compile without syntax errors.')

	def test_pages_use_shared_dashboard_shell(self):
		for page_path in self.page_files:
			content = page_path.read_text(encoding='utf-8')
			self.assertIn('configure_dashboard_page(', content)
			self.assertIn('render_page_header(', content)

	def test_unified_entrypoint_registers_all_pages(self):
		app_path = self.streamlit_root / 'app.py'
		content = app_path.read_text(encoding='utf-8')
		self.assertIn('configure_dashboard_page()', content)
		self.assertIn('st.Page("pages/overview.py"', content)
		self.assertIn('st.Page("pages/realtime_monitor.py"', content)
		self.assertIn('st.Page("pages/model_metrics.py"', content)
		self.assertIn('st.Page("pages/review_queue.py"', content)
		self.assertIn('st.Page("pages/truth_predict_analysis.py"', content)