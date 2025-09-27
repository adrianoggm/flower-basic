#!/usr/bin/env python3
"""Automated ECG5000 to WESAD/SWELL Migration Script.

This script automatically migrates deprecated ECG5000 usage to WESAD and SWELL
datasets. It performs safe replacements with backup creation.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


class ECG5000Migrator:
    """Automated migrator for ECG5000 to WESAD/SWELL datasets."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.backup_dir = project_root / "migration_backups"
        self.migrations_applied = []
        
        # Define migration patterns
        self.migration_patterns = {
            # Import statements
            r'from \.utils import load_ecg5000_openml': 
                'from .datasets import load_wesad_dataset',
            
            r'load_ecg5000_openml': 
                'load_wesad_dataset',
                
            # Function calls with parameters
            r'load_ecg5000_openml\(\)':
                'load_wesad_dataset()',
                
            # Comments and documentation
            r'ECG5000 dataset': 'WESAD dataset',
            r'ecg5000 dataset': 'WESAD dataset', 
            r'datos ECG5000': 'datos WESAD',
            r'ECG5000 locales': 'WESAD locales',
            
            # Model sequence length (ECG5000 specific)
            r'seq_len.*140.*ECG5000': 'seq_len: Variable length based on dataset (WESAD/SWELL)',
            
            # Print statements
            r'Loading ECG5000 dataset': 'Loading WESAD dataset',
            r'📥 Loading ECG5000 dataset': '📥 Loading WESAD dataset',
        }
    
    def create_backup(self, file_path: Path) -> Path:
        """Create backup of file before migration."""
        self.backup_dir.mkdir(exist_ok=True)
        
        # Create backup path with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{file_path.name}_{timestamp}.bak"
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def migrate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Migrate a single file from ECG5000 to WESAD/SWELL."""
        if not file_path.exists() or file_path.suffix != '.py':
            return False, []
            
        # Read original content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        original_content = content
        changes_made = []
        
        # Apply migration patterns
        for pattern, replacement in self.migration_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                changes_made.append(f"Replaced '{pattern}' -> '{replacement}' ({len(matches)} occurrences)")
        
        # Check if changes were made
        if content != original_content:
            # Create backup
            backup_path = self.create_backup(file_path)
            
            # Write migrated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.migrations_applied.append({
                'file': file_path,
                'backup': backup_path,
                'changes': changes_made
            })
            
            return True, changes_made
        
        return False, []
    
    def migrate_project(self, target_dirs: List[str] = None) -> Dict:
        """Migrate entire project from ECG5000 to WESAD/SWELL."""
        if target_dirs is None:
            target_dirs = ['src/flower_basic']
        
        results = {
            'files_processed': 0,
            'files_migrated': 0,
            'total_changes': 0,
            'errors': []
        }
        
        for target_dir in target_dirs:
            dir_path = self.project_root / target_dir
            if not dir_path.exists():
                results['errors'].append(f"Directory not found: {dir_path}")
                continue
            
            # Process all Python files
            for py_file in dir_path.rglob('*.py'):
                # Skip certain files
                if 'test' in py_file.name.lower() or 'deprecated' in py_file.name.lower():
                    continue
                
                results['files_processed'] += 1
                
                try:
                    migrated, changes = self.migrate_file(py_file)
                    if migrated:
                        results['files_migrated'] += 1
                        results['total_changes'] += len(changes)
                        
                except Exception as e:
                    results['errors'].append(f"Error migrating {py_file}: {e}")
        
        return results
    
    def generate_report(self) -> str:
        """Generate migration report."""
        if not self.migrations_applied:
            return "No migrations were applied."
        
        report = ["ECG5000 to WESAD/SWELL Migration Report", "=" * 50, ""]
        
        for migration in self.migrations_applied:
            report.append(f"File: {migration['file']}")
            report.append(f"Backup: {migration['backup']}")
            report.append("Changes:")
            for change in migration['changes']:
                report.append(f"  - {change}")
            report.append("")
        
        report.append(f"Total files migrated: {len(self.migrations_applied)}")
        
        return "\n".join(report)


def main():
    """Main migration script."""
    project_root = Path.cwd()
    migrator = ECG5000Migrator(project_root)
    
    print("Starting ECG5000 to WESAD/SWELL migration...")
    print(f"Project root: {project_root}")
    
    # Run migration
    results = migrator.migrate_project()
    
    # Print results
    print(f"\nMigration completed:")
    print(f"Files processed: {results['files_processed']}")
    print(f"Files migrated: {results['files_migrated']}")
    print(f"Total changes: {results['total_changes']}")
    
    if results['errors']:
        print(f"Errors encountered: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Generate detailed report
    report = migrator.generate_report()
    report_path = project_root / "migration_report.txt"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    if results['files_migrated'] > 0:
        print(f"Backups created in: {migrator.backup_dir}")
        print("\nNOTE: Please review changes and run tests before committing!")


if __name__ == '__main__':
    main()
